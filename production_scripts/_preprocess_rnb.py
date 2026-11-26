# #!/usr/bin/env python3
from production_scripts._preprocess_cadastre_vectors import * 
from pathlib import Path
from production_scripts.download_vectors import _fix_shape_column_for_qgis
from qgis.core import QgsVectorLayer

class PreprocessRNB(VectorsPreprocess):
    """
    Classe de prétraitement des données de cadastre RNB.
    """
    def build_resulting_vectors_file_path(self, unzipped_path : Path) -> Path :
        original_file_path = unzipped_path / rf"cadastre-{self.dept_code}-batiments-csv"/ rf"RNB_{self.dept_code}.csv"
        cleaned_file_path = unzipped_path / rf"cadastre-{self.dept_code}-batiments-csv"/ rf"RNB_{self.dept_code}_reduced.csv"
        if(not cleaned_file_path.exists()):
            if(not original_file_path.exists()):
                raise FileNotFoundError(f"Original RNB file not found: {original_file_path}")
            cleaned_file_path = _fix_shape_column_for_qgis(original_file_path)
        return cleaned_file_path

    def build_version_cadastre(self, date : datetime): 
        return datetime.strftime(date, '%B, %Y, Cadastre R.N.B.')
    
    def update_fields(self, qgis_vec_tools : VectorTools, preprocessed_vector : QgsVectorLayer):
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
        
    def final_check(self, preprocessed_vector : QgsVectorLayer):
        # TODO quelles vérifications pour RNB preprocessing ?
        pass
                   