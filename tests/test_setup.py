#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from production_scripts import download_vectors as dwd 
import shutil
import subprocess
from os import makedirs

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
        
        
def clean_up_test_rasters_images_only():
    ''' supprime les résultats de l'étape split exclusivement'''
    tdir = Path(UNITTESTS_FOLDER_PATH) 
    target_vectors_cadastre_folder_path = tdir / 'split'
    # remove the entire folders
    rm_images_cmd = f'find "{target_vectors_cadastre_folder_path}" -type d -name "images" -exec rm -r {{}} +'
    rm_repart_cmd = f'find "{target_vectors_cadastre_folder_path}" -type d -name "repartition_by_rasters" -exec rm -r {{}} +'  
    rm_otiles_cmd = f'find "{target_vectors_cadastre_folder_path}" -type d -name "only_tiles" -exec rm -r {{}} +'  
    
    # deletes files inside images folders
    # rm_images_cmd = f'find "{target_vectors_cadastre_folder_path}" -type d -name "images" -exec find {{}} -type f -delete \\;' # only files
    # rm_repart_cmd = f'find "{target_vectors_cadastre_folder_path}" -type d -name "repartition_by_rasters" -exec find {{}} -type f -delete \\;' # only files
    # rm_otiles_cmd = f'find "{target_vectors_cadastre_folder_path}" -type d -name "only_tiles" -exec find {{}} -type f -delete \\;' # only files
    
    rm_logs_f_cmd = f'find "{target_vectors_cadastre_folder_path}" -type f \\( -name "progress.txt" -o -name "notes.txt" \\) -delete'
    
    os.system(rm_images_cmd) # expl :  remove all files in /tmp/sgis/unittests/split/09/2026/rasters/images
    os.system(rm_repart_cmd) # expl :  remove all files in /tmp/sgis/unittests/split/09/2026/rasters/repartition_by_rasters
    os.system(rm_otiles_cmd) # expl :  remove all files in /tmp/sgis/unittests/split/09/2026/rasters/only_tiles

    os.system(rm_logs_f_cmd) # remove progress.txt, notes.txt