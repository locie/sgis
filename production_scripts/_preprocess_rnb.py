# #!/usr/bin/env python3
from production_scripts._preprocess_cadastre_vectors import * 
from datetime import datetime
from pathlib import Path

class PreprocessRNB(VectorsPreprocess):
    """
    Classe de prétraitement des données de cadastre RNB.
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
            date = datetime.strptime(self.cadastre_dir, '%Y-%m-%d')
        except ValueError as e:
            print("\n\n Invalid date format. Expected YYYY-MM-DD:")
            raise e

        self.version_cadastre = datetime.strftime(date, '%B, %Y, Cadastre RNB')
        # IO files

        unzipped_folder_path = raw_base_folder/ RELATIVE_VECTORS_CADASTRE_FOLDER_PATH / self.cadastre_dir / "unzipped"
        self.vectors_layer_raw_path = unzipped_folder_path / rf"cadastre-{self.dept_code}-batiments-csv"/ rf"RNB_{self.dept_code}.csv"
        self.preprocessed_buildings_layer_filename = 'batiments.shp' # TODO voir s'il n'est pas préférable de faire l'export CSV au lieu de ShapeFile
        
        # vérification des paramètres
        self.check_params()
    
    def update_fields(self, preprocessed_vector):
        """
        Prepare a preprocessed vector layer for export by removing oversized fields and refactoring field widths.
        Removes RNB-specific fields that exceed the 254 character limit imposed by the Shapefile format.
        Refactors the width of specified fields to comply with Shapefile format constraints (max 254 characters).
        """
         # remove RNB fields exceeding 254 characters which are not supported by shapefile format
        fields_to_remove = ["ext_ids", "addresses", "plots", "status"]
        self.qgis_vec_tools.remove_fields(preprocessed_vector, fields_to_remove) # remove fields
        # refactor fields of width 255 by truncating to 254.
        fields_to_refactor = ["rnb_id", "point"] 
        preprocessed_vector = self.qgis_vec_tools.refactor_field_width(preprocessed_vector, fields_to_refactor, 254) 
        
    def final_check(self, preprocessed_vector):
        # TODO quelles vérifications pour RNB preprocessing ?
        pass
                   