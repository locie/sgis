from concurrent.futures import ThreadPoolExecutor
from math import isinf
from threading import Lock
from time import perf_counter, sleep

from PyQt5.QtCore import QVariant
from pandas import DataFrame
from processing import run
from qgis.core import QgsFeature, QgsField, QgsRasterLayer, QgsVectorLayer
from tqdm import tqdm

from .._utils import get_logger, prepare_paths
from ..vector_tools import load_layer
from .._init_qgis import processing # needed to initialize



class Splitter():
    """Prepare the splitting process of some raster layers according to a vector layer.

    Parameters
    ----------
    vector_layer_path : path-like
        Full path to the *.shp file.

        The layer must have an 'ID' field . Split images are named according to this field.
        Related vector files must have the same name (except file extension (*.dbf, *.prj, *.shx)).
    raster_layers_path : path-like
        Path of the directory containing raster files (*.jp2, *.tiff).
    output_directory_path : path-like
        Path of the directory where split images are saved.

            - if exists, actual content may be overwritten
            
            - if does not exist, it is created (including parents)

    Raises
    ------
    FileNotFoundError
        If any of the files with extension *.dbf, *.prj or *.shx is not found.
    FileNotFoundError
        If `raster_layers_path` does not contain any raster file.


    """
        # fixme: testme: enlever le fichier *.shx et constater l'exception

    def __init__(self, vector_layer_path, raster_layers_path, output_directory_path):
        
        vector_layer_path, raster_layers_path, output_directory_path = prepare_paths(vector_layer_path, raster_layers_path, output_directory_path)
        for file_ext in ('shx', 'prj', 'dbf'): # shx: index, dbf: attributes, prj: projection
            if not vector_layer_path.with_suffix(f'.{file_ext}').exists():
                raise FileNotFoundError(f"File '{vector_layer_path.stem}.{file_ext}' is missing.")
        self._vector_layer_path = vector_layer_path

        raster_files = []
        for file in raster_layers_path.iterdir():
            if file.suffix in ('.jp2', '.tiff'):
                raster_files.append(file)
        if not raster_files:
            raise FileNotFoundError(f'No raster file found in {raster_layers_path}.')
        self._raster_files = raster_files
        self._raster_layers_path = raster_layers_path

        (output_directory_path / 'images').mkdir(parents=True, exist_ok=True)
        (output_directory_path / 'repartition_by_rasters').mkdir(parents=True, exist_ok=True)
        self._output_path = output_directory_path
        self._output_images_path = output_directory_path / 'images'
        self._output_repartition_path = output_directory_path / 'repartition_by_rasters'



    @property
    def vector_layer_path(self):
        return self._vector_layer_path
    
    @property
    def raster_layers_path(self):
        return self._raster_layers_path


    @staticmethod
    def _test_file_exists(folder, filename):
        """
        Check whether the file named `filename` exists in `folder`.

        Returns: bool
        """
        path = folder / filename
        exists = path.exists()
        return exists, path

    @staticmethod
    def _add_suffix(folder, filename):
        """
        Add a suffix to the stem of `filename`.

        :returns the path of the file name with suffix
        """
        stem, file_extension = filename.rsplit('.', 1)
        suffix = 1
        new_filename = f'{stem}_{suffix}.{file_extension}'
        exists, path = Splitter._test_file_exists(folder, new_filename)
        while exists:
            suffix += 1
            new_filename = f'{stem}_{suffix}.{file_extension}'
            exists, path = Splitter._test_file_exists(folder, new_filename)
        return path



    def _determine_unprocessed_rasters(self):
        '''
        Look for a 'progress.txt' file in the output directory. If exists, load the content (already processed rasters).
        If does not exist, create one.

        :return: tuple
            - processed rasters
            - path to the file
        '''
        logger = get_logger()

        input_rasters = self._raster_files
        progress_file = self._output_path / 'progress.txt'

        try:
            with open(progress_file, "r") as f:
                already_processed = [e[:-1] for e in f.readlines()]
        except FileNotFoundError as e:
            logger.info("No existing progress file found, creating one.")
            already_processed = []
            with open(progress_file, "w") as f:
                f.write('')
        already_processed = [r for r in already_processed if r in map(str, input_rasters)]  # only the processed ones that have to do with the currently treated layers
        logger.info(f"{len(already_processed)}/{len(input_rasters)} rasters were previously processed")
        input_rasters = [r for r in input_rasters if str(r) not in already_processed]
        logger.info(f"{len(input_rasters)} rasters remaining")

        return input_rasters, progress_file

    def split(self, threads_num=None, overwrite_with_suffix=True):
        '''
        Create a *.jpg file for every intersection of:

        - the vector layer at `vector_layer_path`
        - the raster layers at `vector_layer_path`

        The image files are stored in `output_directory_path`.
        Multiple rasters can be processed in a concurrent way using the `threads_num` argument.

        Parameters
        ----------
        threads_num: int
            Number of threads to allocate to the splitting task.
            Each thread is given one raster to process at a time.

        overwrite_with_suffix: bool
            Define what to do in case an image file with same name already exists.
            - If True, the new image is saved with a suffix appended to its name.
            Must be used to get all images of buildings that are spread accross several rasters.
            - If False, the new image is not saved.


        Returns
        -------

        '''
        logger = get_logger()
        shapefile = str(self._vector_layer_path)
        input_rasters, progress_file = self._determine_unprocessed_rasters()

        if not overwrite_with_suffix:
            logger.warning('`overwrite_with_suffix=False`: In case of a building spread over several rasters, only one image will be saved.')

        if (threads_num == 1) or (threads_num is None):
            logger.info('Sequential mode')
            for raster in input_rasters:  # tqdm(input_rasters, desc=f'Raster loop', leave=True, colour='green', unit='raster', ncols=100):
                self._find_split_intersect(raster, shapefile, overwrite_with_suffix)
                with open(progress_file, 'a') as f:
                    f.write(str(raster) + '\n')
        else:
            if not isinstance(threads_num, int):
                raise TypeError(f'Invalid type for `threads_num`.')
            logger.info(f"Number of threads: {threads_num}")
            if threads_num > 6:
                logger.warning("The problem is I/O bound: a large number of threads does not largely improve performance.")

            lock_file_write = Lock()
            # [info] la Semaphore doit être appliquée en amont,
            # notamment pour ne pas charger tous les rasters en RAM (début de la méthode `_find_split_intersect`)
            def _threaded(raster, overwrite_with_suffix, lock_file_write):
                self._find_split_intersect(raster, shapefile, overwrite_with_suffix)
                with lock_file_write:
                    with open(progress_file, 'a') as f:
                        f.write(str(raster) + '\n')

            with ThreadPoolExecutor(threads_num) as executor:
                for raster in tqdm(input_rasters, desc=f'Raster loop - {threads_num} threads', leave=True,
                                   miniters=10, colour='green', unit='raster', ncols=100):
                    executor.submit(_threaded, raster, overwrite_with_suffix, lock_file_write)

    


    def _find_split_intersect(self, raster, shapefile, overwrite_with_suffix):
        '''
        1) Compute the OMBB of each feature 2) test whether it intersects with the raster
        '''
        raster_layer = QgsRasterLayer(str(raster), raster.name)
        vector_layer = load_layer(str(shapefile), 'preprocessed_vector')
        logger = get_logger()

        if 'ID' not in vector_layer.fields().names():
            raise KeyError("Vector layer must have an 'ID' attribute.")
        vector_layer_OMBB = QgsVectorLayer('Polygon',
                                           f'temporary_layer_{id(raster_layer)}',
                                           'memory')  # `id(...)` is just used to set a random identifier to make sure memory is not shared due to same name
        pr = vector_layer_OMBB.dataProvider()
        pr.addAttributes([QgsField("ID", QVariant.String)])
        vector_layer_OMBB.updateFields()
        vector_layer_OMBB.setCrs(vector_layer.crs())


        vector_layer_OMBB.startEditing()
        raster_layer_extent = raster_layer.extent()
        data_provider = vector_layer_OMBB.dataProvider()
        for feature in tqdm(vector_layer.getFeatures(), desc=f'Overlap', leave=True, colour='blue', unit='building',
                            total=vector_layer.featureCount(), ncols=100):

            bounding_box = feature.geometry().boundingBox()

            cond = raster_layer_extent.intersect(bounding_box).area()
            if cond and not isinf(cond):
                logger.debug(f"Feature with ID '{feature['ID']}' intersects raster '{raster.name}'") # fixme: uncomment
                polygon = feature.geometry().orientedMinimumBoundingBox()[0]

                ft = QgsFeature()
                ft.setGeometry(polygon)
                ft.setAttributes([feature['ID']])
                data_provider.addFeature(ft)

        vector_layer_OMBB.commitChanges()


        self._define_images_names(
            input_vector=vector_layer_OMBB,
            input_raster=raster_layer,
            overwrite_with_suffix=overwrite_with_suffix
        )

    def _define_images_names(self, input_vector, input_raster, overwrite_with_suffix):
        '''
        Set image names according to the 'ID' file of the vector layer.
        If image already exists, add a suffix to the name.

        Parameters
        ----------
        input_vector
        input_raster

        Returns
        -------


        Notes
        -----
        A building can overlap several raster files. In this case, if no suffix was added to the image name,
        each image would overwrite the previous one.

        '''
        logger = get_logger()
        out_folder = self._output_images_path
        vector_layer = input_vector
        raster_layer = input_raster

        vector_layer.subsetString()

        features = [str(i['ID']) for i in vector_layer.getFeatures()]

        buildings = set(features)
        t0 = perf_counter()
        args = []

        formatter_ = lambda name: '"ID" = \'{}\''.format(name)

        for name in buildings:
            filename = f'{name}.jpg'
            exists, path = self._test_file_exists(out_folder, filename)
            q = formatter_(name)
            if exists:
                path = self._add_suffix(out_folder, filename)

            if ((not exists) or overwrite_with_suffix):
                args.append((q, path, filename))


        # writing process is repeated until all files do exist on disk
        to_write = args.copy()
        count = 0
        raster_name = raster_layer.name()
        while to_write:
            if count >= 1:
                logger.warning(f'{raster_name}: {count} nth attempt to write images:'
                      f'{len(to_write)} buildings remaining')

            # normal disk write
            for q, path, filename in tqdm(to_write, desc=f'Buildings - splitter: {raster_name} ', leave=False,
                                          colour='red', unit='buildings', ncols=150):
                self._do_split(raster_layer, vector_layer, q, path)

            # files existence check
            failed_write = []
            for q, path, filename in to_write:
                if not path.exists():
                    failed_write.append((q, path, filename))

            # prepare for another loop if needed
            to_write = failed_write.copy()
            count += 1

        # improvement of raster tracability: association of image names and the corresponding raster
        # --> the names are the one that should have been written on disk, not the one actually written
        #       --> yet, since the writing process is repeated until success, both lists should be the same
        raster_buildings = DataFrame(args)
        if not raster_buildings.empty:
            raster_buildings.columns = ['q', 'Suffixed file name', 'File name']
            raster_buildings['Suffixed file name'] = raster_buildings['Suffixed file name'].apply(lambda x: x.name)
            raster_buildings = raster_buildings.iloc[:, -2:]
            path_raster_buildings = self._output_repartition_path
            path_raster_buildings.mkdir(exist_ok=True)
            raster_buildings.to_csv(path_raster_buildings / raster_name.replace(".jp2", ".csv"),
                                    index=False)

        t1 = perf_counter()
        if not raster_buildings.empty:
            logger.info(
                f"{raster_name}: {len(args)} buildings took {t1 - t0:1.0f} seconds, i.e. {(t1 - t0) / len(args):1.2f} s/buildings.")
        else:
            logger.info(f"{raster_name}: 0 buildings found.")



    def _do_split(self, raster_layer, vector_layer, q, path):
        '''
        Call the `gdal:cliprasterbymasklayer` function using Qgis `processing`.

        '''
        vector_layer.setSubsetString(q)
        params = {
            'INPUT': raster_layer,
            'OUTPUT': str(path),
            'MASK': vector_layer,
            'ALPHA_BAND': False,
            'CROP_TO_CUTLINE': True,
            'KEEP_RESOLUTION': True,
            'OPTIONS': 'COMPRESS=LZW',
            'DATA_TYPE': 0,
            'MULTITHREADING': False,  # no effect
        }
        run('gdal:cliprasterbymasklayer', params)
