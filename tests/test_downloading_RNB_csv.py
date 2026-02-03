#!/usr/bin/env python3
import unittest
from production_scripts import download_RNB_geojson_csv as rnb


TEST_RNB_URL="https://rnb-opendata.s3.fr-par.scw.cloud/files/RNB_09.csv.zip"
TEST_TARGET_FOLDER="/tmp/"
EXPECTED_SHA1_FOR_TESTED_FILE="4960cf1b8187e618a5fce94fca88102d7eb2ad77"
'''
URL

    https://rnb-opendata.s3.fr-par.scw.cloud/files/RNB_09.csv.zip
URL stable
    https://www.data.gouv.fr/api/1/datasets/r/f2d0ecc2-f4c7-4bcd-be24-dce215e2df1a
Identifiant
    f2d0ecc2-f4c7-4bcd-be24-dce215e2df1a
sha1
    4960cf1b8187e618a5fce94fca88102d7eb2ad77
'''

class WidgetTestCase(unittest.TestCase):
    def dwd_rnb_09(self):
        sha1_of_zip_file =rnb.download_file(TEST_RNB_URL, TEST_TARGET_FOLDER)
        self.assertEqual(sha1_of_zip_file, EXPECTED_SHA1_FOR_TESTED_FILE)
        
if __name__ == '__main__':
    unittest.main()