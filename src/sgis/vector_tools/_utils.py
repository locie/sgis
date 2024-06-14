from qgis.core import (
    QgsVectorLayer,
    QgsCoordinateTransformContext,
    QgsVectorFileWriter
)

from .._init_qgis import processing, feedback, context
from .._utils import get_logger, prepare_paths

def load_layer(full_path, layer_name):
    """Load a vector layer.

    Parameters
    ----------
    full_path : path-like
        Path of the SHP layer to load.
    layer_name : str
        Name of the loaded layer.

    Returns
    -------
    QgsVectorLayer
        Layer at `full_path`.

    Raises
    ------
    ValueError
        Whenever Qgis loading of the layer fails.
    Notes
    -----
    Modifications of the layer may propagate to the original file on disk.

    """
    full_path = prepare_paths(full_path, as_str=True)
    logger = get_logger()
    logger.info(f"Loading layer at {full_path}")
    layer = QgsVectorLayer(full_path, layer_name, "ogr")
    if not layer.isValid():
        raise ValueError('Layer failed to load!')
    else:
        return layer


def copy_layer(layer, name=None):
    """Copy a layer in memory. 
    
    Parameters
    ----------
    layer : QgsVectorLayer
        Layer to copy. The geomtry type must be 'Polygon'.
    name : str, optional
        Name of the copy. If None, name is the name of `layer` appended with '_copy'.

    Returns
    -------
    QgsVectorLayer
        Copied layer
    """
    logger = get_logger()
    logger.info('Copying layer in memory')
    if name is None:
        name = f'{layer.name()}_copy'
    copy = QgsVectorLayer("Polygon", name, "memory")
    copy.setCrs(layer.crs())
    layer_data = layer.dataProvider()
    attr = layer_data.fields()
    feats = layer_data.getFeatures()

    copy_data = copy.dataProvider()
    copy_data.addAttributes(attr)
    copy.updateFields()
    copy_data.addFeatures(feats)
    return copy

def export_csv(layer, full_path):
    """Write the attribute table of a vector layer in a CSV file.

    Parameters
    ----------
    layer : QgsVectorLayer
        Layer whose attributes must be exported.
    full_path : path-like
        Path of the CSV file.

    Raises
    ------
    AttributeError
        If `full_path` is not a path to a CSV file.

    """
    full_path = prepare_paths(full_path, as_str=True)
    if not full_path.endswith('csv'):
        raise AttributeError("`full_path` must be a 'CSV' file.")
    _export(layer, full_path)

def export_shp(layer, path, name):
    """Export a vector layer as an SHP file.

    Parameters
    ----------
    layer : QgsVectorLayer
        Layer to export
    path : path-like
        Path of the directory to which SHP file is exported.
    name : str
        Name of the exported SHP file.

    Raises
    ------
    AttributeError
        If `path` is not a directory.
    """

    # Strange behaviour of writeAsVectorFormatV3, do not modify. Notes about writeAsVectorFormatV3:
    # - if given path is an existing dir, does nothing
    # - else, depends. Sometimes creates a dir at `full_path` (even if `full_path` describes a shp file) and use it to save the vectors files.
    # See also:
    #    https://gis.stackexchange.com/questions/435772/pyqgis-qgsvectorfilewriter-writeasvectorformatv3-export-z-dimension-not-workin
    #    https://gis.stackexchange.com/questions/352506/list-of-usable-ogr-drivers-for-pyqgis

    from pathlib import Path
    path = prepare_paths(path)
    if not path.is_dir():
        raise AttributeError("`path` must be directory.")
    full_path = str(path)
    name = str(name)
    if not name.endswith('.shp'):
        name += '.shp'
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "ESRI Shapefile"
    QgsVectorFileWriter.writeAsVectorFormatV3(layer,
                                              str(Path(full_path, name)),
                                              QgsCoordinateTransformContext(),
                                              options
                                              )


def _export(layer, path):
    processing.run(
        "native:savefeatures",
        {
            'INPUT': layer,
            'OUTPUT': path,
        },
        context=context,
        feedback=feedback,
    )





# Useful Qgis functions:
# # show all processing algorithms
# for alg in QgsApplication.processingRegistry().algorithms():
#         print(alg.id(), "->", alg.displayName())

# # display some help regarding a specific algorithm
# processing.algorithmHelp('native:fieldcalculator'