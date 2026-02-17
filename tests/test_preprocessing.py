#!/usr/bin/env python3
import test_setup # ignore unused import
import unittest # The test framework
from production_scripts import preprocess


class Test_Preprocessing(unittest.TestCase):
      # Before executing this test: rm -r /home/pitardg/split/09
      def test_preprocess_etalab_cadastre_09(self):
            # input params
            dep='09';
            year='2025';
            cadastre_dir='2025-12-01'
            resolution=20;
            print('')
            preprocess.main('etalab',dep, year, cadastre_dir, resolution)
      
      def test_preprocess_rnb_cadastre_09(self):
            # input params
            dep='09';
            year='2026';
            cadastre_dir='2026-02-14'
            resolution=20;
            print('')
            preprocess.main('rnb',dep, year, cadastre_dir, resolution)