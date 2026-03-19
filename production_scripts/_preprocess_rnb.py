# #!/usr/bin/env python3
from production_scripts._preprocess_cadastre_vectors import * 
from pathlib import Path

class PreprocessRNB(VectorsPreprocess):
    """
    Classe de prétraitement des données de cadastre RNB.
    """
    def build_resulting_vectors_file_path(self, unzipped_path : Path) -> Path :
        return unzipped_path / rf"cadastre-{self.dept_code}-batiments-csv"/ rf"RNB_{self.dept_code}.csv"
    
    def build_version_cadastre(self, date : datetime): 
        return datetime.strftime(date, '%B, %Y, Cadastre R.N.B.')
    
    def update_fields(self, qgis_vec_tools : VectorTools, preprocessed_vector):
        """
        Prepare a preprocessed vector layer for export by removing oversized fields and refactoring field widths.
        Removes RNB-specific fields that exceed the 254 character limit imposed by the Shapefile format.
        Refactors the width of specified fields to comply with Shapefile format constraints (max 254 characters).
        """
         # remove RNB fields exceeding 254 characters which are not supported by shapefile format
        fields_to_remove = ["ext_ids", "addresses", "plots", "status"]
        qgis_vec_tools.remove_fields(preprocessed_vector, fields_to_remove) # remove fields
        # refactor fields of width 255 by truncating to 254.
        fields_to_refactor = ["rnb_id", "point"] 
        return qgis_vec_tools.refactor_field_width(preprocessed_vector, fields_to_refactor, 254) 
        
    def final_check(self, preprocessed_vector):
        # TODO quelles vérifications pour RNB preprocessing ?
        pass
                   