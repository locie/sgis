#!/usr/bin/env python3
from os import path
from production_scripts import download_RNB_geojson_csv as rnb # The code to test
# from sgis.vector_tools import QgisManager
import unittest # The test framework

TEST_RNB_URL="https://rnb-opendata.s3.fr-par.scw.cloud/files/"
TEST_TARGET_FOLDER="/tmp/"
EXPECTED_SHA1_FOR_TESTED_FILE_RNB_09="4960cf1b8187e618a5fce94fca88102d7eb2ad77"

class Test_Downloading(unittest.TestCase):
    def test_dowloading_RNB_09_csv(self):
        filename="RNB_09.csv.zip"
        url=TEST_RNB_URL+filename
        target=TEST_TARGET_FOLDER+filename
        
        sha1_of_zip_file =rnb.download_file(url, target)
        # Verifie le sha1 du fichier de test
        self.assertEqual(sha1_of_zip_file, EXPECTED_SHA1_FOR_TESTED_FILE_RNB_09)
        
    def test_getRnbMetadata(self):
        filename="RNB_09.csv.zip"
        url=TEST_RNB_URL+filename
        rnb.get_csv_metadata(url)
        
    def test_unzip(self):
        filename="RNB_09.csv.zip"
        zip_path=TEST_TARGET_FOLDER+filename
        target=TEST_TARGET_FOLDER
        self.assertTrue(path.exists(zip_path), f"{zip_path} not found. Execute once the following test: test_dowloading_RNB_09_csv") 
        rnb.unzip(zip_path, TEST_TARGET_FOLDER)
        unzipped_filename="RNB_09.csv"
        self.assertTrue(path.exists(target+unzipped_filename))
        
    # def test_loadVectorLayerFromGeojson(self):
    #     filename="RNB_09.csv"
    #     geojson_csv_file_path=TEST_TARGET_FOLDER+filename
    #     self.assertTrue(path.exists(geojson_csv_file_path), f"{geojson_csv_file_path} not found. Execute once the following tests: \ntest_dowloading_RNB_09_csv \n test_unzip") 
    #     layername=f'batiments_09'
    #     qjis_mng = QgisManager()
    #     raw_vector = qgis_mng.load_layer(geojson_csv_file_path, layername)
    #     qgis_mng.close()
            
if __name__ == '__main__':
    unittest.main()