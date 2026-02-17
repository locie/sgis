#!/usr/bin/env python3
import test_setup as ts
import unittest # The test framework
from production_scripts.target_folders_manager import TargetFoldersManager

class Test_scan_results(unittest.TestCase):
  def test_scan_results_folders(self):
    dep='09';
    year='2025';
    target_folders_manager = TargetFoldersManager(dep, year)
    # target_folders_manager.scan_target_folders()
    target_folders_manager.scan_cadastre_vectors_folder()