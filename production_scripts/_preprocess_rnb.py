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
        super().__init__(dep) # Appel du constructeur parent
        self.cadastre_dir = cadastre_dir
        self.resolution = resolution
        try:
            date = datetime.strptime(self.cadastre_dir, '%Y-%m-%d')
        except ValueError as e:
            print("\n\n Invalid date format. Expected YYYY-MM-DD:")
            raise e

        self.version_cadastre = datetime.strftime(date, '%B, %Y, Cadastre RNB')
        # IO files
        self.vectors_layer_raw_path = Path(rf'{self.home_path}/LaCie_thebaulm/gis/vectors/cadastre/{self.cadastre_dir}/unzipped/cadastre-{self.dep_code}-batiments-csv/RNB_{self.dep_code}.csv')
        self.preprocessed_buildings_layer_filename = 'batiments.shp' # TODO voir s'il n'est pas préférable de faire l'export CSV au lieu de ShapeFile
        
        # vérification des paramètres
        self.check_params()