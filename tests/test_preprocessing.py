#!/usr/bin/env python3
import test_setup as ts
import unittest 

from production_scripts import preprocess
from production_scripts import download_vectors as dwd 
from pathlib import Path
from datetime import datetime

class Test_Preprocessing(unittest.TestCase):
      
      @classmethod
      def setUpClass(self):
            """
            Configure la classe de test en initialisant les répertoires de test.
            Cette méthode prépare l'environnement de test en :
            1. Création du dossier principal des résultats de test s'il n'existe pas
            2. Suppression de tout dossier de vecteurs de cadastre existant d'une session de test précédente
            La méthode garantit une table rase pour chaque test en effaçant l'ancien cadastre.
            résultats vectoriels tout en préservant la structure du répertoire principal des résultats de test.
            """
            ts.create_tests_results_folder()
            ts.clean_up_test_cadastres_folders()
          
      def test_preprocess_etalab_cadastre_09(self):
            # input params
            t_dep='09';
            t_year='2025';
            t_date='2025-12-01'
            t_cadastre_dir = t_date
            # download
            dwd.main(t_dep,  data_type="etalab", date=t_date, raw_data_folder = ts.UNITTESTS_FOLDER_PATH)
            # preprocess Etalab
            preprocess.main('etalab',dep = t_dep, year = t_year, cadastre_dir = t_cadastre_dir, raw_data_folder = ts.UNITTESTS_FOLDER_PATH,  dest_folder_path = ts.UNITTESTS_FOLDER_PATH)
      
      def test_preprocess_rnb_cadastre_09(self):
            metadata = dwd.main("09", raw_data_folder=ts.UNITTESTS_FOLDER_PATH)
            t_date = datetime.strptime(metadata.date, '%Y-%m-%d')
            t_year = datetime.strftime(t_date, "%Y")
            t_cadastre_dir = metadata.date
            preprocess.main('rnb', metadata.dept_code, year = t_year, cadastre_dir = t_cadastre_dir, raw_data_folder = ts.UNITTESTS_FOLDER_PATH,  dest_folder_path = ts.UNITTESTS_FOLDER_PATH)
      
      
      # @unittest.skip("Temporairement désactivé")      
      # def test_geometries_in_RNB_09_csv(self):
      #       """
      #       Explique warning au chargement du fichier RNB_09.csv : [QGIS] - [Warning] - DelimitedText: 1861 record(s) discarded due to incompatible geometry types
      #       """
      #       import pandas as pd
      #       # Load CSV
      #       TESTED_RNB_FILE_PATH=Path("/home/pitardg/LaCie_thebaulm/gis/vectors/cadastre/2024-06-06/unzipped/cadastre-09-batiments-csv/RNB_09.csv")
      #       df = pd.read_csv(TESTED_RNB_FILE_PATH, sep=";")

      #       geometry_column_name = "shape"
      #       pattern = r"^(?:POLYGON|MULTIPOLYGON)"
      #       countnotpolygon = (~df[geometry_column_name].str.contains(pattern, na=False)).sum()
      #       countpoint = (df[geometry_column_name].str.contains("POINT", na=False)).sum()
      #       print()
      #       print("Nombre de lignes ne contenant PAS POLYGON ou MULTIPOLYGON :", countnotpolygon)
      #       print("Nombre de lignes contenant POINT :", countpoint)
            
      #       # save csv file containing only POINT geometries to compare with Cadastre Etalab
      #       df_filtered = df[df[geometry_column_name].str.contains("POINT", na=False)]
      #       new_path = TESTED_RNB_FILE_PATH.parent
      #       new_path /= "FilteringPointFrom_RNB_09.csv"
      #       df_filtered.to_csv(new_path, index=False)