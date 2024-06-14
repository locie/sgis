from functools import partial
from pathlib import Path
from pickle import dump, load, UnpicklingError
from shutil import copy2
import urllib

from numpy import where, exp, sum
from pandas import DataFrame
from tensorflow import cast, where, dtypes
from tensorflow.image import resize_with_crop_or_pad
from tensorflow.keras import losses, Sequential, Model
from tensorflow.keras.applications.efficientnet import EfficientNetB2
from tensorflow.keras.backend import epsilon
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dropout, Dense, RandomBrightness, RandomContrast, \
    Input
from tensorflow.keras.models import load_model as load_model_keras
from tensorflow.keras.optimizers import Adam
from tensorflow.math import count_nonzero, argmax
from tqdm import tqdm

from ._custom_data_loader import image_dataset_from_directory
from .._utils import get_logger, prepare_paths


def create_model(name, img_size=(250, 250), class_names=('no_PV', 'PV'), data_augmentation=True):
    """Create a new CNN classification model. 

    Parameters
    ----------
    name : str
        Name of the model. Will be used when model is exported to disk.
    img_size : tuple of int, optional
        Size of the image fed to to the network. Too large images are cropped. Too small images are padded with 0 (black). By default (250, 250).
        Higher `img_size` values typically perform best but lead to a sharp increase in running time.
    class_names : tuple, optional
        Names of the classes, by default ('no_PV', 'PV'). If `class_names` is of size 2, binary classification is performed, else categorical classification is used.
    data_augmentation : bool, optional
        Whether to increase the diversity of images fed to the network during training and validation.
        If True (default), random brightness (factor=0.125) and random contrat operations (factor=0.875) are applied to each image.
        Data augmentation has no effect on the number of images used.

    Returns
    -------
    CNNModel
        Created model.

    Notes
    -----

    """
    CNNModel_ = CNNModel()
    CNNModel_._name = name
    CNNModel_._img_size = img_size
    CNNModel_._class_names = class_names
    CNNModel_._data_augmentation = data_augmentation
    CNNModel_._set_class_properties(class_names)
    CNNModel_._create_model(data_augmentation)
    return CNNModel_

def load_model(metadata_path, model_path): # full model path is required because it can be an intermediate model saved `fit`
    """Load a previously created CNN model.

    Parameters
    ----------
    metadata_path : path-like
        Path to the *.metadata file.
    model_path : path-like
        Path to the *.keras file.

    Returns
    -------
    CNNModel
        Loaded model.

    """
        # fixme HERE: going upward 
    CNNModel_ = CNNModel()
    logger = get_logger()
    model_path = prepare_paths(model_path)
    if not model_path.suffix == '.keras':
        raise ValueError('Model path must end with a `.keras` suffix.')

    logger.info('Loading model and metadata')

    CNNModel_._load_metadata(metadata_path)
    metrics_ = [CNNModel_._get_precision(class_name) for class_name in CNNModel_.class_names] + \
               [CNNModel_._get_recall(class_name)    for class_name in CNNModel_.class_names]
    custom_objects = {func.__name__: func for func in metrics_}
    CNNModel_._model = load_model_keras(model_path, custom_objects=custom_objects)
    return CNNModel_

class CNNModel():
    _num_channels = 3  # todo: enable user modification?
    _transform = lambda height, width: partial(resize_with_crop_or_pad,
                                               target_height=height,
                                               target_width=width)

    @property
    def name(self):
        return self._name

    @property
    def img_size(self):
        return self._img_size
    @property
    def class_names(self):
        return self._class_names

    @property
    def model(self):
        return self._model
    

    def _set_class_properties(self, class_names):
        self._num_classes = len(class_names)
        self._class_names = class_names
        self._class_names_idx = dict(enumerate(class_names))
        self._reverse_class_names_idx = {v: k for k, v in self._class_names_idx.items()}
        self._mode = 'BINARY' if self._num_classes == 2 else 'CATEGORICAL'
        self._label_mode = self._mode.lower()

    def _get_precision(self, class_name):
        def precision(y_true, y_pred):
            """Precision metric.
            Only computes a batch-wise average of precision.
            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            class_id = self._reverse_class_names_idx[class_name]
            if self._mode == 'CATEGORICAL':
                y_pred = argmax(y_pred, axis=1)  # `y_pred` of size (BATCH_SIZE, n_classes)
                y_true = argmax(y_true, axis=1)  # `y_true` of size (BATCH_SIZE, n_classes)
            else:
                y_pred = y_pred[:, 0]
                y_true = y_true[:, 0]                    # `y_true` of size (BATCH_SIZE, 1)
                y_pred = where(y_pred > 0.5, 1, 0)  # threshold is 0.5 if output of RNN is the output of sigmoid
            y_true = cast(y_true, dtypes.float64)
            y_pred = cast(y_pred, dtypes.float64)
            cond_predicted_positives = y_pred == class_id
            cond_correct_predictions = y_pred == y_true
            predicted_positives = count_nonzero(cond_predicted_positives, dtype=dtypes.float64)
            true_positives = count_nonzero(cond_correct_predictions & cond_predicted_positives,
                                                   dtype=dtypes.float64)
            precision_ = true_positives / (predicted_positives + epsilon())
            return precision_

        precision.__name__ = f'Precision (class: {class_name})'
        return precision

    def _get_recall(self, class_name):
        def recall(y_true, y_pred):
            """Recall metric.
            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            Only computes a batch-wise average of recall.
            """
            class_id = self._reverse_class_names_idx[class_name]
            if self._mode == 'CATEGORICAL':
                y_pred = argmax(y_pred, axis=1)
                y_true = argmax(y_true, axis=1)
            else:
                y_pred = y_pred[:, 0]
                y_true = y_true[:, 0]
                y_pred = where(y_pred > 0.5, 1, 0)

            y_true = cast(y_true, dtypes.float64)
            y_pred = cast(y_pred, dtypes.float64)
            cond_predicted_positives = y_pred == class_id
            cond_actual_positives = y_true == class_id
            cond_correct_predictions = y_pred == y_true
            true_positives = count_nonzero(cond_correct_predictions & cond_predicted_positives,
                                                   dtype=dtypes.float64)
            actual_positives = count_nonzero(cond_actual_positives, dtype=dtypes.float64)
            recall_ = true_positives / (actual_positives + epsilon())
            return recall_

        recall.__name__ = f'Recall (class: {class_name})'
        return recall
    
    def _create_model(self, data_augmentation):
        logger = get_logger()
        from_logits = False
        if self._mode == 'BINARY':

            loss = losses.BinaryCrossentropy(
                    from_logits=from_logits)
            nbr_units = 1
            activation = 'sigmoid'
        else:
            loss = losses.CategoricalCrossentropy(
                    from_logits=from_logits)
            nbr_units = self._num_classes
            activation = 'softmax'

        base_model = EfficientNetB2(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=self.img_size + (CNNModel._num_channels,)
        )

        base_model.trainable = True
        inputs = Input(shape=self.img_size + (CNNModel._num_channels,))
        if data_augmentation:
            logger.warning('`data_augmentation`=True might slow down computation')
            data_augmentation = [
                RandomBrightness(
                    factor=0.125,  # f in [-factor, factor], img += f*255
                ),
                RandomContrast(factor=0.875)
            ]
            data_augmentation = Sequential(data_augmentation)
            inputs_efficient_net = data_augmentation(inputs)
        else:
            inputs_efficient_net = inputs

        outputs = base_model(inputs_efficient_net)
        outputs = Dropout(0.3)(outputs)
        outputs = Dense(nbr_units,
                        activation=activation,
                        bias_initializer=None               # fixme
                        )(outputs)

        self._model = Model(inputs=inputs, outputs=outputs)


        metrics_ = [self._get_precision(class_name) for class_name in self.class_names] + \
                   [self._get_recall(class_name)    for class_name in self.class_names]

        self.model.compile(
            optimizer=Adam(),
            loss=loss,
            metrics=metrics_
        )


    def load_fit_datasets(self, training_path, training_share, validation_share, validation_path=None, batch_size=32):
        """Load some images to train and validate the model.

        Parameters
        ----------
        training_path : path-like
            Path of a directory containing labelled images in dedicated directories, named according to `class_names`.
        training_share : float
            Must satisfy `0 < training_share < 1`.   Proportion of the images to be used for training.
        validation_share : float
            Must satisfy `0 < validation_share < 1`. Proportion of the images to be used for validation.
        validation_path : path-like, optional
            If provided, target to the validation data set. If None, data in `training_path` is split into a training and a validation subsets. By default None.
        batch_size : int, optional
            Number of images processed at a time, by default 32.

        Raises
        ------
        ValueError
            If `training_share + validation_share <= 1`.

        """
        logger = get_logger()
        logger.info('Loading training and validation datasets')
        if validation_path is None:
            validation_path = training_path
        training_path, validation_path = prepare_paths(training_path, validation_path)
        if not isinstance(training_share, float) and (0<training_share<1):
            raise ValueError('`training_share` must satisfy 0<`training_share`<1')
        if not isinstance(validation_share, float) and (0<validation_share<1):
            raise ValueError('`validation_share` must satisfy 0<`validation_share`<1')
        # if not isinstance(batch_size, int):                       # fixme: instead, should be stated in the documentation
        #     raise TypeError('`batch_size` must be an integer')

        transform = CNNModel._transform(self.img_size[0], self.img_size[1])

        kwargs = dict(label_mode=self._label_mode,
                      class_names=self.class_names,
                      shuffle=True,
                      seed=42,
                      batch_size=batch_size,
                      image_size=self.img_size,
                      transform=transform)
        if training_path == validation_path:
            if training_share + validation_share > 1:
                raise ValueError('`training_share` + `validation_share` <= 1 is needed if same dir is used for both')
            logger.info(f"Using directory '{training_path}' for both the training and the validation sets. Sets are disjoint.")
            self._fit_dataset_path = training_path
            share = training_share + validation_share
            validation_share_in_training = validation_share / training_share
            self._training_set = image_dataset_from_directory(
                directory=training_path,
                subset='training',
                validation_split=1 - (1 - validation_share_in_training) * share,
                **kwargs
            )
            self._val_set = image_dataset_from_directory(
                directory=training_path,
                subset='validation',
                validation_split=validation_share_in_training * share,
                **kwargs

            )
            fp_training = set(self._training_set.file_paths)
            fp_validation = set(self._val_set.file_paths)
            assert not fp_training & fp_validation  # datasets are disjoints
        else:
            self._fit_dataset_path = (training_path, validation_path)
            self._training_set = image_dataset_from_directory(
                directory=training_path,
                subset='training',
                validation_split=1 - training_share,
                **kwargs
            )
            self._val_set = image_dataset_from_directory(
                directory=validation_path,
                subset='validation',
                validation_split=validation_share,
                **kwargs
            )
            # datasets are disjoints only if the content of the two directories are disjoint

    def fit(self, epochs, lr_init, lr_callback=None, auto_save=None, initial_epoch=0):       
        """Train the model using the fitting subset of the training dataset.

        Must be called after `load_fit_datasets`.

        Parameters
        ----------
        epochs : int
            Number of epochs to train the model. 
            An epoch is an iteration over the entire data provided. 
            Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". 
            The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached. (excerpt from Keras documentation)
        lr_init : float
            Learning rate at epoch 0.
        lr_callback : keras.callbacks.Callback, optional
            Callback that modifies the learning rate depending on the epoch, by default None.
        auto_save : path-like, optional
            Path of a directory where the model is saved at the end of every epoch. A suffix is appended to the model name to describe the epoch. By default None.
        initial_epoch : int, optional
            Epoch at which to start training, by default 0.

        Notes
        -----
        The learning process is performed using an Adam optimizer and a cross-entropy loss function.
        """
        logger = get_logger()
        try:
            self._training_set
        except AttributeError as e:
            raise AttributeError('`load_fit_datasets` must be called first')
        if not (isinstance(initial_epoch, int) and initial_epoch <= epochs - 1):
            raise ValueError('`initial_epoch` must satisfy  `initial_epoch` <= `epochs` - 1')
        callbacks = []
        if auto_save is not None:
            auto_save = prepare_paths(auto_save)
            if not auto_save.is_dir():
                raise ValueError('`auto_save` must be a directory path')
            logger.info('Model is saved at the end of every epoch, with a dedicated name suffix.')
            path = prepare_paths(auto_save / self.name, as_str=True) + r'_{epoch:03d}.keras'
            callbacks += [ModelCheckpoint(filepath=path,
                                          save_weights_only=False,
                                          # monitor='val_loss',
                                          # mode='min',
                                          save_freq='epoch',
                                          save_best_only=False)]
            self._export_metadata(auto_save)

        if lr_callback is not None:
            callbacks += [lr_callback]


        self.model.optimizer.learning_rate.assign(lr_init)
        self.model.fit(
            x=self._training_set,
            epochs=epochs,
            verbose=2,
            callbacks=callbacks,
            initial_epoch=initial_epoch,                # todo: disable user modification?
            validation_data=self._val_set,
        )

    def evaluate(self, batch_size=32):
        """Assesses the performances of the model using the validation subset of the training dataset.

        Must be called after `load_fit_datasets`.

        Parameters
        ----------
        batch_size : int, optional
            Number of images processed at a time, by default 32.

        Returns
        -------
        List of floats
            Loss and metric values.
                   

        Return a dict of values: loss function and metrics
        """
        logger = get_logger()
        logger.info('Model evaluation begins')
        result = self.model.evaluate(
            x=self._val_set,
            batch_size=batch_size,
            verbose=2,
            return_dict=True,
        )
        return result


    def _load_prediction_dataset(self, dataset_path, share):
        """ 
        :parameter share: float in ]0, 1[, default 1. Percentage of data that is loaded from the disk.
        :parameter transform: transformation applied to each image of the dataset.

        

        Parameters
        ----------
        dataset_path : path-like
            Must contain an 'images' directory.
        share : float
             in ]0, 1[]. Percentage of data that is loaded from the disk.
        """
        logger = get_logger()
        logger.debug(f"Loading {share:.3%} of images stored in '{dataset_path}'")
        transform = CNNModel._transform(self.img_size[0], self.img_size[1])
        self._prediction_set = image_dataset_from_directory(
            label_mode=None,
            directory=dataset_path,
            shuffle=False,
            subset='validation',
            validation_split=share,
            seed=42,
            batch_size=32,              # todo: enable user modification?
            image_size=self.img_size,
            transform=transform,
        )
    def predict(self, input_path, output_path, share=0.999, copy_images=False, save_scores=True):
        """Apply the model to images stored locally. 
        
        In 2 classes mode, the prediction score is compared to the 0.5 threshold in order to determine the class.
        In 3 classes mode, the maximum value of the 3 scores defines the class. A softmax normalization is performed on scores.

        Parameters
        ----------
        input_path : path-like
            Path of a directory containing the images to apply the model on. An intermediate `images` directory is required:
            `input_path` > 'images' > some images
        output_path : path-like
            Results directory. Created if does not exist.
        share : float, optional
            Share of the images to be used. Must satisfy `0<share<1`. Default to 0.999.
            Note that due to the use of some Keras methods, `share=1` is not a valid call. In that case, please set `share=1-eps` with eps a small number.
        copy_images : bool, optional

            Whether to write the predicted images (copy) on disk.

                - if False (default), images are not saved

                - if True, each image is copied in a directory "PV" (prediction >= threshold) or "no_PV" (prediction < threshold).

        save_scores : bool, optional
            
            If True (default), the following are exported in a file 'scores.csv':
                
                - prediction value (in ]0, 1[, rounded 7f)
                - classification results

        Returns
        -------
        pandas.DataFrame
            Results of the model, identical to those saved with `save_scores = True`.

        Notes
        -----
        Building having a name like '\d+_\d.jpg' (regex-like) are ignored. Those buildings are at raster borders. 


        
        """
                # fixme: testme: mode 3 classes

        if not any(save_scores, copy_images):
            raise ValueError(f"Both `save_scores` and `copy_images` are set to False which would lead to no exported results.")
        logger = get_logger()
        input_path, output_path = prepare_paths(input_path, output_path)
        if not (input_path / 'images').exists():
            raise FileNotFoundError(f"`input_path` must have a subdirectory 'images' containing images, got '{input_path}'.")
        if not isinstance(share, float) and ((0 < share) and (share <1)):
            raise ValueError('Share must satisfy 0 < `share` < 1.')
        self._load_prediction_dataset(input_path, share)
        logger.info('Predictions begin')
        predictions = self._model.predict(x=self._prediction_set, verbose=2)

        logger.info('Predictions postprocessing')
        if self._mode == 'BINARY':
            predicted_class = where(predictions > 0.5, 1, 0)[:, 0]
            predictions_score = predictions
        else:
            predicted_class = predictions.argmax(axis=1)
            predictions_exp = exp(predictions)
            predictions_score = predictions_exp / sum(predictions_exp, axis=1, keepdims=True)
        df = DataFrame(data=predictions_score.max(axis=1), columns=['Score']).round(7)
        df['Predicted class'] = predicted_class
        df['Predicted class'] = df['Predicted class'].replace(self._class_names_idx)
        df['File paths'] = self._prediction_set.file_paths
        df['Building name'] = df['File paths'].apply(lambda p: Path(p).stem)
        df = df[~df['Building name'].str.contains('\d_\d{1,2}', regex=True)]
        df = df[~df['Building name'].str.contains('.', regex=False)]
        if copy_images:
            logger.info('Saving images')
            df_ = df.copy()
            col = 'Predicted class'
            [(output_path / p).mkdir(exist_ok=True, parents=True) for p in df_[col].unique()]
            for _, (building_name, dest_dir_) in \
                    tqdm(df_[['Building name', col]].iterrows(), desc='Writing predicted images', leave=True,
                         colour='green', unit='images', ncols=100):
                savedir = output_path / dest_dir_ / f'{building_name}.jpg'
                copy2(src=input_path / 'images' / f'{building_name}.jpg',
                         dst=savedir)

        df = df.set_index('Building name')[['Score', 'Predicted class']]
        if save_scores:
            logger.info('Saving prediction scores')
            output_path.mkdir(exist_ok=True, parents=True)
            filename = output_path / 'scores.csv'
            df.to_csv(filename, float_format='%.7f')

        return df

    def predict_from_coordinates(self, input_save_path, output_path, min_x, min_y, max_x, max_y, resolution=0.2, copy_images=False, save_scores=True):
        """For the French case only. Apply the classification model to raw tiles downloaded from the IGN database.

        An arbitrary large area is described using x and y coordinates.
        The corresponding height and width depend on the resolution. This image is split into smaller pieces that
        match the CNN network input shape.


        Parameters
        ----------
        input_save_path : path-like
            Directory where downloaded images are saved. Created if does not exist.
        output_path : path-like
            Directory where prediction results are saved. Created if does not exist.
        min_x : float
            Left coordinate of the box of the prediction area, given in EPSG:3857.
        min_y : float
            Bottom coordinate of the box of the prediction area, given in EPSG:3857.
        max_x : float
            Right coordinate of the box of the prediction area, given in EPSG:3857.
        max_y : float
            Top coordinate of the box of the prediction area, given in EPSG:3857.
        resolution : float, optional
            meter / pixel. Used to infer image height and width from coordinates. By default 0.2
        copy_images : bool, optional
            Whether to write the predicted images (copy) on disk in the directory `output_path`.

            - if False (default), images are not saved

            - if True, each image is copied in directories according to class names

        save_scores : bool, optional
            If True (default), both prediction value (in ]0, 1[, rounded 7f)
            and classification results ({0, 1}) are exported in a csv file in `output_path`.

        Returns
        -------
        pandas.DataFrame
            Prediction results.


        Notes
        -----
        1. Querying images outside of France will return blank images. 
        
        2. In all cases, the following must be satisfied:

            - `minx >= -20037508`
            - `maxx <= 20037508` 
            - `miny >= -30240971`
            - `maxy <= 30240971`

        3. Some online tools provide box coordinates using manual selection, for instance http://bboxfinder.com.

        4. The used IGN database is a WMS raster flux, with layer name "HR.ORTHOIMAGERY.ORTHOPHOTOS". 
        It describes the BD ORTHO® V3 database, i.e. aerial imagery at resolution 20 cm. An internet connection is required.

        """

        # problème: toutes les images ne sont pas prédites: problème du share < 1
        # fixme: 1) choix1 : redéfinir les méthodes Keras de vérification
        #        2) choix 2: construire manuellement un dataset TensorFlow
        # assez problématique, même pour load_fit_datasets


        logger = get_logger()
        input_save_path, output_path = prepare_paths(input_save_path, output_path)
        input_save_path = input_save_path / 'images'
        input_save_path.mkdir(exist_ok=True, parents=True)

        # converting the (x, y) coordinates to (width, height) based on the resolution
        img_height, img_width = self.img_size
        DX = (max_x - min_x)
        DY = (max_y - min_y)
        given_img_height = DY / resolution
        given_img_width = DX / resolution
        if given_img_height <= img_height:
            used_img_height = given_img_height
            logger.info(f'`min_y`, `max_y` and `resolution` correspond to only one image along the height dimension (due to network input height being {img_height}).')
        else:
            used_img_height = img_height
            if given_img_height % img_height:
                logger.info(f'`min_y`, `max_y` and `resolution` correspond to several images along the height dimension (due to network input height being {img_height}).')
                shiftY = ((given_img_height % img_height) / 2) * resolution
                min_y += shiftY
                max_y -= shiftY

        if given_img_width <= img_width:
            used_img_width = given_img_width
            logger.info(f'`min_x`, `max_x` and `resolution` correspond to only one image along the height dimension (due to network input width being {img_width}).')
        else:
            used_img_width = img_width
            if given_img_width % img_width:
                logger.info(f'`min_x`, `max_x` and `resolution` correspond to several images along the height dimension (due to network input width being {img_width}).')
                shiftX = ((given_img_width % img_width) / 2) * resolution
                min_x += shiftX
                max_x -= shiftX



        # assert not width % img_width          # note: pas exact, mais peu importe car les images sont redimensionnées avant de rentrer dans le réseau
        # assert not height % img_height

        # defining sub tiles from the larger given tile
        tiles = {}
        dx = used_img_width * resolution
        dy = used_img_height * resolution
        y = min_y
        name_y = 0
        while y < max_y:
            x = min_x
            name_x = 0
            while x < max_x:
                name = f'{name_x}-{name_y}'
                tiles[name] = (x, y, x + dx, y + dy)
                x += dx
                name_x += 1
            y += dy
            name_y += 1

        # downloading sub tiles using the IGN API
        remaining_tiles = tiles.copy()
        for _ in range(5):
            if remaining_tiles:
                for name, (min_x_, min_y_, max_x_, max_y_) in tqdm(tiles.items(), desc='Downloading images from IGN', leave=True,
                                                                    colour='green', unit='images', ncols=100):
                    if name in remaining_tiles:
                        url = \
                           r'https://data.geopf.fr/wms-r?' \
                            'LAYERS=HR.ORTHOIMAGERY.ORTHOPHOTOS' \
                            '&FORMAT=image/jpeg' \
                            '&SERVICE=WMS&VERSION=1.3.0' \
                            '&REQUEST=GetMap&STYLES' \
                            '&CRS=EPSG:3857' \
                           f'&WIDTH={used_img_width}' \
                           f'&HEIGHT={used_img_height}' \
                           f'&BBOX={min_x_},{min_y_},{max_x_},{max_y_}'
                        try:
                            urllib.request.urlretrieve(url, input_save_path / f'{name}.jpg')
                            remaining_tiles.pop(name)
                        except BaseException as e:
                            if not isinstance(e, urllib.error.HTTPError):
                                pass
                            logger.warn(
                                f'Unable to retrieve image with coordinates {(min_x_, min_y_, max_x_, max_y_)}. Will try again a few times.')
                            logger.debug(f'The following exception was raised: {e}')
                            logger.debug(f'url is {url}')
        if remaining_tiles:
            logger.error(f'{len(remaining_tiles)} out of {len(tiles)} images could not be downloaded. Skipping these images.')
        return self.predict(input_save_path.parent, output_path, copy_images=copy_images, save_scores=save_scores)

    def save_model(self, path):
        """Save the model and its metadata.

        Parameters
        ----------
        path : path-like
            Path of a directory to save the model. The model itself is saved as a *.keras file. Its metadata are saved as a *.metadata binary file.

        Notes
        -----
        The metadata file contains some important information to load the model.
        """
        logger = get_logger()
        logger.info('Saving model and metadata')
        path = prepare_paths(path)
        model_path = path / f'{self.name}.keras'
        self.model.save(model_path, overwrite=True)
        self._export_metadata(path)

    def _export_metadata(self, path):
        metadata = (self.name, self.img_size, self.class_names)
        metadata_path = path/f'{self.name}.metadata'
        f = open(metadata_path, 'wb')
        dump(metadata, f)
        f.close()


    def _load_metadata(self, metadata_path):
        f = open(metadata_path, 'rb')
        try:
            self._name, self._img_size, class_names = load(f)
        except UnpicklingError as e:
            raise UnpicklingError("`metadata_path` must point to the '[...].metadata' file created by the `save_model` method.")
        self._set_class_properties(class_names)
        f.close()