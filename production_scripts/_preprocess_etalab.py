# #!/usr/bin/env python3
from production_scripts._preprocess_cadastre_vectors import *
from qgis.core import QgsVectorLayer
from pathlib import Path

class PreprocessEtalab(VectorsPreprocess):
    """
    Classe de prétraitement des données de cadastre d'Etalab.
    """
    def build_resulting_vectors_file_path(self, unzipped_path : Path) -> Path :
        return unzipped_path / f"cadastre-{self.dept_code}-batiments-shp/batiments.shp"
    
    def build_version_cadastre(self, date : datetime):
        return datetime.strftime(date, '%B, %Y, Cadastre Etalab')
    
    def update_fields(self, qgis_vec_tools : VectorTools, preprocessed_vector):   
        """
        Add ID
        """ 
        return qgis_vec_tools.add_ID(preprocessed_vector, self.prefix)
        
    def final_check(self, preprocessed_vector : QgsVectorLayer):
        # vérification que les attributs sont les bons, par exemple code dep sur 3 digits
        attributes = preprocessed_vector.getFeature(500).attributeMap()
        if attributes['ID'][:3] != self.prefix:
                attr = attributes['ID'][:3]
                raise NameError(f'Bad prefix for images. Expected {self.prefix} got {attr}.')