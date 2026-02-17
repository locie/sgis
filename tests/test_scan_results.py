#!/usr/bin/env python3
import test_setup as ts
import unittest # The test framework
from production_scripts.scan_io_folders import scan_cadastre_vectors_folder

class Test_scan_results(unittest.TestCase):
  def test_scan_etalab_cadastre_vectors_folder(self):
    scan_cadastre_vectors_folder("etalab")
    
  def test_scan_rnb_cadastre_vectors_folder(self):
    scan_cadastre_vectors_folder("rnb")