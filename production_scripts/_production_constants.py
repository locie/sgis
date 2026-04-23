RAW_FOLDERNAME="LaCie_thebaulm"

# Downloading
CHUNCK_SIZE=8192
BASE_ETALAB_URL=" https://files.data.gouv.fr/cadastre/etalab-cadastre" # ETALAB
BASE_RNB_URL="https://rnb-opendata.s3.fr-par.scw.cloud/files" # R.N.B. : Référentiel National des Bâtiments

# Preprocessing
RELATIVE_VECTORS_CADASTRE_FOLDER_PATH=rf"gis/vectors/cadastre"
RELATIVE_TEMP_TILES_FOLDER_PATH=rf"temporary_LaCie/rasters/only_tiles"  
PREPROCESSED_BUILDINGS_LAYER_FILENAME="batiments.shp"

RESOLUTION=20
BUFFER_DISTANCE_M=4 # Distance tampon à ajouter autour des bâtiments, en mètres
MIN_AREA_M2=10 # Seuil de suppression des petits bâtiments, en m²

# Classifier
DEFAULT_CLASSIFIER_SHARE_VALUE = 0.99999
DEFAULT_CLASSIFIER_MODEL = '2C_22_34_35_67_73__V14'
DEFAULT_CLASSIFIER_GPU = 'true'