#!/usr/bin/env python3
import sys
from pathlib import Path
from production_scripts import download_vectors as dwd 
import shutil
from os import makedirs, path

UNITTESTS_FOLDER_PATH="/tmp/sgis/unittests/"
TEST_RNB_URL="https://rnb-opendata.s3.fr-par.scw.cloud/files/"
TEST_ARIEGE_09_FILENAME="RNB_09.csv.zip" #Ariege
TEST_RNB_09_URL=TEST_RNB_URL+TEST_ARIEGE_09_FILENAME
TEST_RNB_09_ZIP_TARGET_PATH=UNITTESTS_FOLDER_PATH+TEST_ARIEGE_09_FILENAME

# Add src folders to sys.path once for all tests
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "production_scripts"))

def create_tests_results_folder():    
    ''' cree le dossier pour contenir les résultats de unittests'''
    tdir = Path(UNITTESTS_FOLDER_PATH) 
    if(not tdir.exists()):
        makedirs(tdir)

def clean_up_test_cadastres_folders():
    ''' supprime les résultats d'une session de tests précédente'''
    tdir = Path(UNITTESTS_FOLDER_PATH) 
    
    target_vectors_cadastre_folder_path = tdir / dwd.RELATIVE_VECTORS_CADASTRE_FOLDER_PATH  
    target_rasters_cadastre_folder_path = tdir / 'split'         
    if(target_vectors_cadastre_folder_path.exists()):
        # supprime tout le contenu de /tmp/sgis/unittests/gis/vectors/cadastre
        shutil.rmtree(target_vectors_cadastre_folder_path)
    if(target_rasters_cadastre_folder_path.exists()):
        # supprime tout le contenu de /tmp/sgis/unittests/split/
        shutil.rmtree(target_rasters_cadastre_folder_path)
    
    # supprime le fichier /tmp/sgis/unittests/RNB_09.csv.zip uniquement
    target_rnb_09_csv_path = Path(TEST_RNB_09_ZIP_TARGET_PATH)
    if(target_rnb_09_csv_path.exists()):
        target_rnb_09_csv_path.unlink(missing_ok=False)   