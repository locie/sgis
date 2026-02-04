#!/usr/bin/env python3
'''
cas d'absence de session X (i.e. pas de support Qt)
solution: déclarer une variable d'env:
      os.environ["QT_QPA_PLATFORM"] = "offscreen"
''' 
from os import environ
environ["QT_QPA_PLATFORM"] = "offscreen"

from production_scripts import preprocess# The code to test
import unittest # The test framework

class Test_Preprocessing(unittest.TestCase):
      def test_preprocessCadastre09(self):
            # input params
            dep='09';
            year='2025';
            cadastre_dir='2025-12-01'
            resolution=20;

            preprocess.main(dep, year, cadastre_dir, resolution)
            # TODO check target