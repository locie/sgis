#!/usr/bin/env python3
import test_setup as ts
import unittest # The test framework
from production_scripts.split import main

class Test_splitter(unittest.TestCase):
    def test_split_rasters_from_etalab_buildings(self):
        #   rm -r /tmp/sgis/unittests/split/
        #   cp -rs /tmp/sgis/unittests/temporary_LaCie/only_tiles/09/2025/09-2025-0M20-RGB/. ./

        t_dep='09'
        t_year='2025'
        t_resolution=20
        t_threads_num=6
        main(t_dep, t_year, t_resolution, t_threads_num, raw_folder_path=ts.UNITTESTS_FOLDER_PATH, dest_folder_path=ts.UNITTESTS_FOLDER_PATH)

    def test_split_rasters_from_RNB_buildings(self):
        #   rm -r /tmp/sgis/unittests/split/
        #   cp -rs /tmp/sgis/unittests/temporary_LaCie/only_tiles/09/2025/09-2025-0M20-RGB/. /tmp/sgis/unittests/temporary_LaCie/rasters/only_tiles/09/2024/09-2024-0M20-RGB/
        t_dep='09'
        t_year='2024'
        t_resolution=20
        t_threads_num=6
        main(t_dep, t_year, t_resolution, t_threads_num, raw_folder_path=ts.UNITTESTS_FOLDER_PATH, dest_folder_path=ts.UNITTESTS_FOLDER_PATH)
        
        # bash command :
        # python -m unittest discover -s tests -k split_rasters