#!/usr/bin/env python3
from production_scripts.rnb_geo_api import extract_rnb_sha1
import test_setup # ignore unused import
import unittest # The test framework
from pathlib import Path
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
            
      def test_geometries_in_RNB_09_csv(self):
            """
            Explique warning au chargement du fichier RNB_09.csv : [QGIS] - [Warning] - DelimitedText: 1861 record(s) discarded due to incompatible geometry types
            """
            import pandas as pd
            # Load CSV
            TESTED_RNB_FILE_PATH=Path("/home/pitardg/LaCie_thebaulm/gis/vectors/cadastre/2026-02-14/unzipped/cadastre-09-batiments-csv/RNB_09.csv")
            df = pd.read_csv(TESTED_RNB_FILE_PATH, sep=",")

            # Replace 'geometry_column' with the name of your WKT column
            geometry_column_name = "shape"
            pattern = r"^(?:POLYGON|MULTIPOLYGON)"
            countnotpolygon = (~df[geometry_column_name].str.contains(pattern, na=False)).sum()
            countpoint = (df[geometry_column_name].str.contains("POINT", na=False)).sum()
            print()
            print("Nombre de lignes ne contenant PAS POLYGON ou MULTIPOLYGON :", countnotpolygon)
            print("Nombre de lignes contenant POINT :", countpoint)
            
            # save csv file containing only POINT geometries to compare with Cadastre Etalab
            df_filtered = df[df[geometry_column_name].str.contains("POINT", na=False)]
            new_path = TESTED_RNB_FILE_PATH.parent
            new_path /= "FilteringPointFrom_RNB_09.csv"
            df_filtered.to_csv(new_path, index=False)

      def test_get_SHA1_checksum(self):
            print()
            import requests

            # Correct dataset JSON
            url = "https://www.data.gouv.fr/api/1/datasets/referentiel-national-des-batiments/"
            data = requests.get(url).json()

            rnb_data_gouv=[]
            # Iterate over resources
            for r in data.get("resources", []):
                  if r.get("checksum"):
                        # sha1_line = r.get("extras", {}).get("analysis:checksum")
                        sha1_line = r.get("checksum").get("value")
                        if sha1_line:
                              new_data = extract_rnb_sha1(r["url"], sha1_line)
                              if new_data:
                                    rnb_data_gouv.append(new_data)
                             
            rnb_data_gouv.sort(key=lambda x: x[0])  
            
            for e in rnb_data_gouv:
                  print(e) 
      
      
      @unittest.skip("Temporairement désactivé car test long")
      def test_download_all_departements(self):
            # TODO
            pass