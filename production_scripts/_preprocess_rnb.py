# #!/usr/bin/env python3
from production_scripts._preprocess_cadastre_vectors import VectorsPreprocess
from datetime import datetime
from pathlib import Path

class PreprocessRNB(VectorsPreprocess):
    """
    Classe de prétraitement des données de cadastre RNB.
    """
    def __init__(self, dep, year, cadastre_dir, resolution=20):
        self.year = year
        self.cadastre_dir = cadastre_dir
        self.resolution = resolution
        
        super().__init__(dep) # Appel du constructeur parent
        
        # IO files
        self.vectors_layer_raw_path = Path(rf'{self.home_path}/LaCie_thebaulm/gis/vectors/cadastre/{self.cadastre_dir}/unzipped/cadastre-{self.dep_code}-batiments-shp/batiments.shp')
        self.preprocessed_buildings_layer_filename = 'batiments.shp'
        
    def check_params(self): #  checks input parameters before preprocessing
        assert ((len(self.dep_code)==2) or (len(self.dep_code)==3 and self.dep[0]=='9'))
        assert self.resolution in range(1, 100)
        try:
            date = datetime.strptime(self.cadastre_dir, '%Y-%m-%d')
        except ValueError as e:
            print("\n\n Invalid date format. Expected YYYY-MM-DD:")
            raise e

        self.version_cadastre = datetime.strftime(date, '%B, %Y, Cadastre Etalab')
        
        if not Path(self.vectors_layer_raw_path).exists():
            raise FileNotFoundError(f'Etalab cadastre data not found in {self.vector_layer_raw_path.parent()}')