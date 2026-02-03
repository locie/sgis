from production_scripts import download_from_RNB_api as rnbapi# The code to test
import unittest # The test framework

class Test_RNB_api(unittest.TestCase):
      def test_RnbApiLimitation(self):
        dept_code=75
        limit=101
        bbox='5.33, 45.00, 7.15, 46.00' 
        # Probleme : Limitation 100 batiments par requete
        rnbapi.get_dept_buildings_geojson(dept_code) 
        rnbapi.get_bbox_buildings_geojson(bbox, limit)