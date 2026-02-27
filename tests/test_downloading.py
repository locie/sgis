#!/usr/bin/env python3
import test_setup as ts
import unittest 

from production_scripts import download_vectors as dwd
from sgis.vector_tools import VectorTools
from production_scripts.rnb_geo_api import find_dept_metadata, request_all_rnb_csv_metadata
from pathlib import Path



class Test_Downloading(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        """
        Configure la classe de test en initialisant les répertoires de test.
        Cette méthode prépare l'environnement de test en :
        1. Création du dossier principal des résultats de test s'il n'existe pas
        2. Suppression de tout dossier de vecteurs de cadastre existant d'une session de test précédente
        La méthode garantit une table rase pour chaque test en effaçant l'ancien cadastre.
        résultats vectoriels tout en préservant la structure du répertoire principal des résultats de test.
        
        3. Télécharge le fichier de test vers /tmp/sgis/unittests/RNB_09.csv.zip
        """
        ts.create_tests_results_folder()
        ts.clean_up_test_cadastres_folders()
        self.expected_file_path = ts.UNITTESTS_FOLDER_PATH + "RNB_09.csv"
        self.downloaded_rnb_09_csv_sha1 = dwd.download_file(ts.TEST_RNB_09_URL, ts.TEST_RNB_09_ZIP_TARGET_PATH)
    
    def test_check_downloaded_zip_sha1(self):
        all_dept_metadata = request_all_rnb_csv_metadata()
        dep_code=9 #ariege
        ariege_metadata = find_dept_metadata(all_dept_metadata, dep_code)
        expected_sha1 = ariege_metadata.sha1
        self.assertEqual(self.downloaded_rnb_09_csv_sha1, expected_sha1)
    
    def test_unzip(self):
        dwd.unzip(ts.TEST_RNB_09_ZIP_TARGET_PATH, ts.UNITTESTS_FOLDER_PATH)
        self.assertTrue(Path(self.expected_file_path).exists())    
        
    def test_download_rnb_cadastre_09(self):
        dwd.main("09", raw_data_folder = ts.UNITTESTS_FOLDER_PATH)
        
    def test_download_etalab_cadastre_09(self):
        dwd.main("09",  data_type="etalab", date="2025-12-01", raw_data_folder=ts.UNITTESTS_FOLDER_PATH)
                
    def test_load_vectors_layer_from_geojson(self):
        layername=f'batiments_09'
        qjis_vec_tools = VectorTools()
        raw_vector = qjis_vec_tools.load_layer(self.expected_file_path, layername)
        
    @unittest.skip("Temporairement désactivé car test long")
    def test_download_cadastre_01_to_08(self):
        dwd.main("rnb","01","2025-12-01")
        dwd.main("rnb","02","2025-12-01")
        dwd.main("rnb","03","2025-12-01")
        dwd.main("rnb","04","2025-12-01")
        dwd.main("rnb","05","2025-12-01")
        dwd.main("rnb","06","2025-12-01")
        dwd.main("rnb","07","2025-12-01")
        dwd.main("rnb","08","2025-12-01")
            
if __name__ == '__main__':
    unittest.main()