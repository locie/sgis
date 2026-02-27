# #!/usr/bin/env python3
from production_scripts._preprocess_cadastre_vectors import *
from datetime import datetime
from pathlib import Path

class PreprocessEtalab(VectorsPreprocess):
    """
    Classe de prétraitement des données de cadastre d'Etalab.
    """
    def __init__(self, dep, year, cadastre_dir, resolution = RESOLUTION, raw_base_folder = None, dest_folder_path = None):
        
        if(raw_base_folder == None):
            raw_base_folder = Path(environ['HOME']) / RAW_FOLDERNAME
        else:
            raw_base_folder = Path(raw_base_folder)
                  
        self.year = year
        super().__init__(dep, dest_folder_path = dest_folder_path) # Appel du constructeur parent
        self.cadastre_dir = cadastre_dir
        self.resolution = resolution
        try:
            self.date = datetime.strptime(self.cadastre_dir, '%Y-%m-%d')
        except ValueError as e:
            print("\n\n Invalid date format. Expected YYYY-MM-DD:")
            raise e
        
        # IO files
        unzipped_path = raw_base_folder / RELATIVE_VECTORS_CADASTRE_FOLDER_PATH / self.cadastre_dir / "unzipped"
        self.vectors_layer_raw_path = unzipped_path / f"cadastre-{self.dept_code}-batiments-shp/batiments.shp"
        self.preprocessed_buildings_layer_filename = 'batiments.shp'
        self.version_cadastre = datetime.strftime(self.date, '%B, %Y, Cadastre Etalab')
        # vérification des paramètres
        self.check_params()
        
    def update_fields(self, preprocessed_vector):   
        """
        Add ID
        """ 
        return self.qgis_vec_tools.add_ID(preprocessed_vector, self.prefix)
        
    def final_check(self, preprocessed_vector):
        # vérification que les attributs sont les bons, par exemple code dep sur 3 digits
        attributes = preprocessed_vector.getFeature(500).attributeMap()
        if attributes['ID'][:3] != self.prefix:
                attr = attributes['ID'][:3]
                raise NameError(f'Bad prefix for images. Expected {self.prefix} got {attr}.')