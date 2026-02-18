#!/usr/bin/env python3
import test_setup as ts
import unittest # The test framework
from production_scripts import rnb_geo_api as rnbapi # The code to test
import production_scripts.rnb_geo_api as trackchanges

class Test_rnb_api(unittest.TestCase):
  def test_rnb_api_limit(self):
    dept_code=75
    limit=101
    bbox='5.33, 45.00, 7.15, 46.00' 
    # Probleme : Limitation 100 batiments par requete
    rnbapi.get_dept_buildings_geojson(dept_code) 
    rnbapi.get_bbox_buildings_geojson(bbox, limit)
  
  @unittest.skip("Temporarily disabled") 
  def test_track_changes_since_date(self):
    since="2025-10-04T13:28:38Z"
    all_changes_df = trackchanges.get_rnb_modifications(since)
    print(f"Nombre de modifications depuis {since} : {len(all_changes_df)}")
  
  # Fonctionne mais trop lente : 1 requete par commune
  # def test_track_changes_for_all_departments(self):
  #   since="2026-01-04T13:28:38Z"
  #   results = get_rnb_modifications_for_all_departments(since)
  #   print(f"Nombre de departements traites : {len(results)}")
  #   filename="changes_by_dept.txt"
  #   changes_csv_file_path=ts.TEST_TARGET_FOLDER+filename
  #   with open("resultats.txt", "a", encoding="utf-8") as f:
  #     for dep, df in results:
  #       print(f"Nombre de modifications pour le departement {dep} : {len(df)}")
  #       f.write(f"Nombre de modifications pour le departement {dep} : {len(df)}\n")
  
  @unittest.skip("Temporarily disabled")
  def test_track_changes_for_1_departement(self):
    dept_code="02"
    since="2025-10-04T13:28:38Z"
    all_changes_df = trackchanges.get_rnb_modifications_by_departement(dept_code,since)
    print(f"Nombre total de modifications pour le departement {dept_code} depuis {since} : {len(all_changes_df)}")
    unique_rnb_id = all_changes_df['rnb_id'].unique()
    print(f"Soient {len(unique_rnb_id)} batiments concernés : {unique_rnb_id}")