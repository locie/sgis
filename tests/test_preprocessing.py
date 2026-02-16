#!/usr/bin/env python3
import test_setup # ignore unused import
import unittest # The test framework
from production_scripts import preprocess_etalab# The code to test

class Test_Preprocessing(unittest.TestCase):
      def test_preprocess_etalab_cadastre_09(self):
            # input params
            dep='09';
            year='2025';
            cadastre_dir='2025-12-01'
            resolution=20;

            preprocess_etalab.main(dep, year, cadastre_dir, resolution)
            # TODO check target