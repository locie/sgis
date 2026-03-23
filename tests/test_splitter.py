#!/usr/bin/env python3
import test_setup as ts
from pathlib import Path
import unittest # The test framework
from production_scripts.split import main
from sgis.splitter._splitter import check_images_counts

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
        ##  COPIER MANUELLEMENT LES 8 PREMIERES TUILES :
        #   mkdir -p /tmp/sgis/unittests/split/09/2026/rasters/only_tiles/09-2026-0M20-RGB/
        #   ls ~/temporary_LaCie/rasters/only_tiles/09/2026/09-2026-0M20-RGB/ | head -n 16 | xargs -I {} cp -rs ~/temporary_LaCie/rasters/only_tiles/09/2026/09-2026-0M20-RGB/{} /tmp/sgis/unittests/temporary_LaCie/rasters/only_tiles/09/2026/09-2026-0M20-RGB/
    
        t_dep='09'
        t_year='2026'
        t_resolution=20
        t_threads_num=6
        main(t_dep, t_year, t_resolution, t_threads_num, raw_folder_path=ts.UNITTESTS_FOLDER_PATH, dest_folder_path=ts.UNITTESTS_FOLDER_PATH)
        
        # bash command :
        # python -m unittest discover -s tests -k split_rasters
        
    def test_split_edit_final_notes(self):
        rasters_folder_path = Path("/tmp/sgis/unittests/split/09/2026/rasters/");
        check_images_counts(rasters_folder_path)