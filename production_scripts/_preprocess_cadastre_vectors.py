from abc import ABC, abstractmethod
from pathlib import Path
from os import environ
from sgis.vector_tools import VectorTools
from sys import gettrace as sys_gettrace
from datetime import datetime
from _production_constants import *

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
      
      def __init__(self, dept_code, cadastre_dir, resolution = RESOLUTION, raw_folder_path = None, dest_folder_path = None):
             # vérifie formatage cadastre_dir YYYY-MM-DD
            try:
                  date = datetime.strptime(cadastre_dir, '%Y-%m-%d')
            except ValueError as e:
                  print("\n\n Invalid date format. Expected YYYY-MM-DD:")
                  raise e 
            
            # paramètres d'entrée communs à Etalab et RNB
            self.year = str(date.year)
            self.dept_code = dept_code
            self.cadastre_dir = cadastre_dir
            self.resolution = resolution 
            
            #  construit les chemins d'entree/sortie
            if(raw_folder_path == None):
                  home_path = Path(environ['HOME'])
                  raw_folder_path = home_path / RAW_FOLDERNAME
                  dest_folder_path = home_path
            else:
                  raw_folder_path = Path(raw_folder_path)
                  dest_folder_path =  Path(dest_folder_path)
            
            self.version_cadastre = self.build_version_cadastre(date)
            
            if len(dept_code)==2:
                  self.prefix = f'0{dept_code}'
            else:
                  self.prefix = dept_code

            #  chemin des données sources (unzipped)
            unzipped_path = raw_folder_path / RELATIVE_VECTORS_CADASTRE_FOLDER_PATH / self.cadastre_dir / "unzipped"
            self.vectors_layer_raw_path = self.build_resulting_vectors_file_path(unzipped_path)
            
            # chemins de destination des données prétraitées
            self.output_dir_path = dest_folder_path / "split" / self.dept_code/ self.year
            self.output_rasters_dir_path = self.output_dir_path / "rasters"
            self.output_vector_layer_dir_path =  self.output_rasters_dir_path / "preprocessing" / "vectors"  
            
            # chemin du dossier temmporaire pour les rasters de tuiles (créées par preprocess.py, supprimées à la fin du script de détection)
            self.raster_layers_dir_path = dest_folder_path / RELATIVE_TEMP_TILES_FOLDER_PATH / self.dept_code / self.year / rf"{self.dept_code}-{self.year}-0M{self.resolution}-RGB"
            self.create_output_dirs()
            self.check_params()
            
            
            
            
      def check_params(self): #  checks input parameters before preprocessing
        assert ((len(self.dept_code)==2) or (len(self.dept_code)==3 and self.dep[0]=='9'))
        assert self.resolution in range(1, 100)
        if not self.vectors_layer_raw_path.parent.exists():
            raise FileNotFoundError(f'Cadastre data not found in {self.vectors_layer_raw_path.parent}')
      
      def create_output_dirs(self):
            rasters_path = Path(self.raster_layers_dir_path)
            if(not rasters_path.exists()):
                  rasters_path.mkdir(parents=True)
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
            # crée une instance de VectorTools
            with VectorTools() as qgis_vec_tools:
                  if(sys_gettrace() is not None):
                        # en mode debug uniquement -> on active la capture des messages QGIS
                        qgis_vec_tools.catch_qgis_messages_enable()
                  
                  raw_vector = qgis_vec_tools.load_layer(self.vectors_layer_raw_path, f'batiments_{self.dept_code}')
                  raw_vector = qgis_vec_tools.copy_layer(raw_vector)
                  preprocessed_vector, unwanted_buildings_number, initial_buildings_number = qgis_vec_tools.remove_small_features(raw_vector, MIN_AREA_M2)
                  preprocessed_vector = qgis_vec_tools.add_buffer_distance(preprocessed_vector, BUFFER_DISTANCE_M)
                  preprocessed_vector = qgis_vec_tools.add_XY_coordinates(preprocessed_vector)
                  preprocessed_vector = self.update_fields(qgis_vec_tools, preprocessed_vector)
                  qgis_vec_tools.export_shp(preprocessed_vector, self.output_vector_layer_dir_path, PREPROCESSED_BUILDINGS_LAYER_FILENAME)

                  with open(f'{self.output_dir_path}/version_cadastre', 'w') as f:
                        f.write(self.version_cadastre)
                  with open(f'{self.output_dir_path}/version_BDORTHO', 'w') as f:
                        version_BDORTHO = f'{self.year}, BDOrtho database, IGN (RGB, resolution {self.resolution}cm)'
                        f.write(version_BDORTHO)
                  
                  self.final_check(preprocessed_vector)
            
      @abstractmethod # méthode définie dans les classes fille PreprocessEtalab et PreprocessRNB   
      def build_resulting_vectors_file_path(self, unzipped_path : Path) -> Path :
            return 
      
      @abstractmethod # méthode définie dans les classes fille PreprocessEtalab et PreprocessRNB   
      def build_version_cadastre(self, date : datetime):
            return 
      
      @abstractmethod # méthode définie dans les classes fille PreprocessEtalab et PreprocessRNB   
      def update_fields(self, qgis_vec_tools : VectorTools, preprocessed_vector):
            return 
      
      @abstractmethod # méthode définie dans les classes fille PreprocessEtalab et PreprocessRNB   
      def final_check(self, preprocessed_vector):
            pass