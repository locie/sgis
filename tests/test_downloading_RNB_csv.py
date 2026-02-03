#!/usr/bin/env python3

from production_scripts import download_RNB_geojson_csv as rnb # The code to test
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
        
if __name__ == '__main__':
    unittest.main()