#!/usr/bin/env python3
import test_setup as ts # ignore unused import
import unittest # The test framework
from os import path
from production_scripts import download_vectors as rnb # The code to test
from sgis.vector_tools import VectorTools

TEST_RNB_URL="https://rnb-opendata.s3.fr-par.scw.cloud/files/"

class Test_Downloading(unittest.TestCase):
    def test_download_zip(self):
        filename="RNB_09.csv.zip"
        url=TEST_RNB_URL+filename
        target= ts.TEST_TARGET_FOLDER+filename
        sha1_of_zip_file =rnb.download_file(url, target)
        print(f"sha1: {sha1_of_zip_file}")
    
    def test_unzip(self):
        filename="RNB_09.csv.zip"
        zip_path=ts.TEST_TARGET_FOLDER+filename
        target=ts.TEST_TARGET_FOLDER
        self.assertTrue(path.exists(zip_path), f"{zip_path} not found. Execute once the following test: test_dowloading_RNB_09_csv") 
        rnb.unzip(zip_path, ts.TEST_TARGET_FOLDER)
        unzipped_filename="RNB_09.csv"
        self.assertTrue(path.exists(target+unzipped_filename))    
        
    def test_download_rnb_cadastre_09(self):
        rnb.main("rnb","09")
        
    def test_download_etalab_cadastre_09(self):
        rnb.main("etalab","09","2025-12-01")
    
    @unittest.skip("Temporairement désactivé car test long")
    def test_download_cadastre_01_to_08(self):
        rnb.main("rnb","01","2025-12-01")
        rnb.main("rnb","02","2025-12-01")
        rnb.main("rnb","03","2025-12-01")
        rnb.main("rnb","04","2025-12-01")
        rnb.main("rnb","05","2025-12-01")
        rnb.main("rnb","06","2025-12-01")
        rnb.main("rnb","07","2025-12-01")
        rnb.main("rnb","08","2025-12-01")
        
    def test_read_metadata(self):
        filename="RNB_09.csv.zip"
        url=TEST_RNB_URL+filename
        rnb.get_csv_metadata(url)   
        
    def test_load_vectors_layer_from_geojson(self):
        filename="RNB_09.csv"
        geojson_csv_file_path=ts.TEST_TARGET_FOLDER+filename
        self.assertTrue(path.exists(geojson_csv_file_path), f"{geojson_csv_file_path} not found. Execute once the following tests: \ntest_dowloading_RNB_09_csv \n test_unzip") 
        layername=f'batiments_09'
        qjis_vec_tools = VectorTools()
        raw_vector = qjis_vec_tools.load_layer(geojson_csv_file_path, layername)
            
if __name__ == '__main__':
    unittest.main()