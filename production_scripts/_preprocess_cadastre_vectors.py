from abc import ABC, abstractmethod
from pathlib import Path
from os import environ
from sgis.vector_tools import VectorTools
from sys import gettrace as sys_gettrace

BASE_DIR="LaCie_thebaulm" # Nom du dossier de stockage des donnees d'entree et de sortie à la racine du home de l'utilisateur
RESOLUTION=20
BUFFER_DISTANCE_M=4 # Distance tampon à ajouter autour des bâtiments, en mètres
MIN_AREA_M2=10 # Seuil de suppression des petits bâtiments, en m²

class VectorsPreprocess(ABC): # Classe abstraite
      """
      Cette classe gère la préprocessing des données vectorielles du cadastre
      - ajout d’une distance tampon autour des bâtiments (4 m par défaut)
      - suppression des petits bâtiments (emprise au sol < 10 m² par défaut)

      Les données produites sont stockées localement.
      """
      
      vectors_layer_raw_path : str
      version_cadastre : str
      year : str
      preprocessed_buildings_layer_filename : str
      
      def __init__(self, dep_code, resolution = RESOLUTION):
            
            # crée une instance de VectorTools
            self.qgis_vec_tools = VectorTools()
            
            # paramètres d'entrée communs à Etalab et RNB
            self.dep_code = dep_code
            self.home_path = environ['HOME']
            self.resolution = resolution
            if len(dep_code)==2:
                  self.prefix = f'0{dep_code}'
            else:
                  self.prefix = dep_code
            
            # chemins de destination des données prétraitées
            self.output_dir_path = rf'{self.home_path}/split/{self.dep_code}/{self.year}'
            self.output_rasters_dir_path = rf'{self.output_dir_path}/rasters'
            self.output_vector_layer_dir_path =  rf'{self.output_dir_path}/preprocessing/vectors'  
            self.create_output_dirs()
            
            # chemin du dossier temmporaire pour les rasters de tuiles (créées par preprocess.py, supprimées à la fin du script de détection)
            self.raster_layers_dir_path = rf'{self.home_path}/temporary_LaCie/rasters/only_tiles/{self.dep_code}/{self.year}/{self.dep_code}-{self.year}-0M{self.resolution}-RGB'
            
            
      def check_params(self): #  checks input parameters before preprocessing
        assert ((len(self.dep_code)==2) or (len(self.dep_code)==3 and self.dep[0]=='9'))
        assert self.resolution in range(1, 100)
        if not Path(self.vectors_layer_raw_path).exists():
            raise FileNotFoundError(f'Cadastre data not found in {self.vector_layer_raw_path.parent()}')
      
      def create_output_dirs(self):
            try:
                  Path(self.output_rasters_dir_path).mkdir(parents=True)
                  Path(self.output_vector_layer_dir_path).mkdir(parents=True)
            except FileExistsError as e:
                  raise e
            
      def run(self):
            """
            Execute the cadastre vector preprocessing pipeline.
            1. Loads the raw vector layer from the specified path
            2. Creates a copy of the raw vector layer
            3. Removes small features (buildings) below the minimum area threshold
            4. Adds a buffer distance to the remaining features
            5. Assigns unique IDs
            6. Adds X and Y coordinate columns to the attribute table
            7. Exports the preprocessed vector layer to a shapefile format
            8. Writes version information for cadastre and BDORTHO data to files
            """
            
            if(sys_gettrace() is not None):
                  # en mode debug uniquement -> on active la capture des messages QGIS
                  self.qgis_vec_tools.catch_qgis_messages_enable()
            
            raw_vector = self.qgis_vec_tools.load_layer(self.vectors_layer_raw_path, f'batiments_{self.dep_code}')
            raw_vector = self.qgis_vec_tools.copy_layer(raw_vector)
            preprocessed_vector, unwanted_buildings_number, initial_buildings_number = self.qgis_vec_tools.remove_small_features(raw_vector, MIN_AREA_M2)
            preprocessed_vector = self.qgis_vec_tools.add_buffer_distance(preprocessed_vector, BUFFER_DISTANCE_M)
            preprocessed_vector = self.qgis_vec_tools.add_XY_coordinates(preprocessed_vector)
            preprocessed_vector = self.update_fields(preprocessed_vector)
            self.qgis_vec_tools.export_shp(preprocessed_vector, self.output_vector_layer_dir_path, self.preprocessed_buildings_layer_filename)

            with open(f'{self.output_dir_path}/version_cadastre', 'w') as f:
                  f.write(self.version_cadastre)
            with open(f'{self.output_dir_path}/version_BDORTHO', 'w') as f:
                  version_BDORTHO = f'{self.year}, BDOrtho database, IGN (RGB, resolution {self.resolution}cm)'
                  f.write(version_BDORTHO)
            
            self.final_check(preprocessed_vector)
            
      
      @abstractmethod # méthode définie dans les classes fille PreprocessEtalab et PreprocessRNB   
      def update_fields(self, preprocessed_vector):
            return 
      
      @abstractmethod # méthode définie dans les classes fille PreprocessEtalab et PreprocessRNB   
      def final_check(self, preprocessed_vector):
            pass
                      