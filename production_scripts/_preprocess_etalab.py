# #!/usr/bin/env python3
from production_scripts._preprocess_cadastre_vectors import VectorsPreprocess
from datetime import datetime
from pathlib import Path
from sgis.vector_tools import VectorTools

class PreprocessEtalab(VectorsPreprocess):
    """
    Classe de prétraitement des données de cadastre d'Etalab.
    """
    def __init__(self, dep, year, cadastre_dir, resolution=20):
        self.year = year
        super().__init__(dep) # Appel du constructeur parent
        self.cadastre_dir = cadastre_dir
        self.resolution = resolution
        try:
            self.date = datetime.strptime(self.cadastre_dir, '%Y-%m-%d')
        except ValueError as e:
            print("\n\n Invalid date format. Expected YYYY-MM-DD:")
            raise e
        
        # IO files
        self.vectors_layer_raw_path = Path(rf'{self.home_path}/LaCie_thebaulm/gis/vectors/cadastre/{self.cadastre_dir}/unzipped/cadastre-{self.dep_code}-batiments-shp/batiments.shp')
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