#!/usr/bin/env python3
from production_scripts.postprocess import main
import test_setup as ts
import unittest # The test framework


class Test_postprocess(unittest.TestCase):  
    # mkdir -p /tmp/sgis/unittests/split/82/  
    # xargs -I {} cp -rs ~/split/82/{} /tmp/sgis/unittests/split/
    def test_postprocess(self):
        dep="82"
        year="2025"
        epoch=5
        roof_type=None
        merge_overlapping="pv"
        export_shp="false"
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