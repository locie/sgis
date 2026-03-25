#!/usr/bin/env python3
from production_scripts.postprocess import main
import test_setup as ts
import unittest # The test framework


class Test_postprocess(unittest.TestCase):       
    def test_postprocess(self):
        dep="09"
        year="2026"
        epoch=5
        roof_type="true"
        merge_overlapping="pv"
        export_shp="true"
        protected_buildings = None #TODO test example
        name = None #TODO test example
       
        main(dep,
             year, 
             name,
             roof_type, 
             protected_buildings, 
             merge_overlapping, 
             export_shp, 
             epoch, 
             base_folder_path=ts.UNITTESTS_FOLDER_PATH)