#!/usr/bin/env python3
from sgis.splitter._splitting_recap import SplittingRecap
import test_setup as ts
from pathlib import Path
import unittest # The test framework
from production_scripts.split import main

class Test_splitter(unittest.TestCase):
    # def test_split_rasters_from_etalab_buildings(self):
    #     #   rm -r /tmp/sgis/unittests/split/
    #     #   cp -rs /tmp/sgis/unittests/temporary_LaCie/only_tiles/09/2025/09-2025-0M20-RGB/. ./

    #     t_dep='09'
    #     t_year='2025'
    #     t_resolution=20
    #     t_threads_num=6
    #     main(t_dep, t_year, t_resolution, t_threads_num, raw_folder_path=ts.UNITTESTS_FOLDER_PATH, dest_folder_path=ts.UNITTESTS_FOLDER_PATH)

    def test_split_rasters_from_RNB_buildings(self):
        ##  COPIER MANUELLEMENT LES 8 PREMIERES TUILES :
        #   mkdir -p /tmp/sgis/unittests/split/09/2026/rasters/only_tiles/09-2026-0M20-RGB/
        #   ls ~/temporary_LaCie/rasters/only_tiles/09/2026/09-2026-0M20-RGB/ | head -n 16 | xargs -I {} cp -rs ~/temporary_LaCie/rasters/only_tiles/09/2026/09-2026-0M20-RGB/{} /tmp/sgis/unittests/temporary_LaCie/rasters/only_tiles/09/2026/09-2026-0M20-RGB/
        ts.clean_up_test_rasters_images_only()
        t_dep='09'
        t_year='2026'
        t_resolution=20
        t_threads_num=6
        main(t_dep, t_year, t_resolution, t_threads_num, raw_folder_path=ts.UNITTESTS_FOLDER_PATH, dest_folder_path=ts.UNITTESTS_FOLDER_PATH)
        
        # errors reached : 
        # Buildings - splitter: 09-2026-0585-6225-LA93-0M20-E080.jp2 :  40%|████████████████▎                        | 3903/9782 [42:28<28:55,  3.39buildings/s]
        # corrupted double-linked lister: 09-2026-0585-6225-LA93-0M20-E080.jp2
        
        # bash command :
        # python -m unittest discover -s tests -k split_rasters 2>&1 | tee test2.log
        
        # Enable GDAL error messages:
        # CPLSetConfigOption("CPL_DEBUG", "ON");
        # Or run with:
        # export CPL_DEBUG=ON
        
    def test_multi(self):
        ##  COPIER MANUELLEMENT LES 8 PREMIERES TUILES :
        #   mkdir -p /tmp/sgis/unittests/split/09/2026/rasters/only_tiles/09-2026-0M20-RGB/
        #   ls ~/temporary_LaCie/rasters/only_tiles/09/2026/09-2026-0M20-RGB/ | head -n 16 | xargs -I {} cp -rs ~/temporary_LaCie/rasters/only_tiles/09/2026/09-2026-0M20-RGB/{} /tmp/sgis/unittests/temporary_LaCie/rasters/only_tiles/09/2026/09-2026-0M20-RGB/
        # ts.clean_up_test_rasters_images_only()
        # sudo find /tmp  -path '*/sgis*' -prune -o \ -user pitardg -group pitardg -exec rm -rf {} +
        t_dep='09'
        t_year='2026'
        t_resolution=20
        t_threads_num=6
        test_2_folder = "/tmp/sgis2/unittests/"
        main(t_dep, t_year, t_resolution, t_threads_num, raw_folder_path=ts.UNITTESTS_FOLDER_PATH, dest_folder_path=test_2_folder)
        
        
    def test_split_edit_final_notes(self):
        rasters_folder_path = Path("/tmp/sgis2/unittests/split/09/2026/")
        s = SplittingRecap(rasters_folder_path)
        s.summarize()