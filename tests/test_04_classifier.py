#!/usr/bin/env python3
from production_scripts.predict_from_images import main
import test_setup as ts
import unittest # The test framework


class Test_classifier(unittest.TestCase):
    def test_classifier(self):
        dep="09";
        year="2026";
        epoch=5;
        main(dep, year, epoch)
        
    def test_postprocess(self):
        #TODO
        pass