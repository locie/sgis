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
        
    def check_params(self): #  checks input parameters before preprocessing
        assert ((len(self.dep_code)==2) or (len(self.dep_code)==3 and self.dep[0]=='9'))
        assert self.resolution in range(1, 100)
        if not Path(self.vectors_layer_raw_path).exists():
            raise FileNotFoundError(f'Etalab cadastre data not found in {self.vector_layer_raw_path.parent()}')
        
    def export_vectors(self, preprocessed_vector, output_vector_layer_dir_path, preprocessed_buildings_layer_filename):
        qgis_vec_tools.export_shp(preprocessed_vector, self.output_vector_layer_dir_path, self.preprocessed_buildings_layer_filename)