#!/usr/bin/env python3
import argparse
from datetime import datetime
import sys
from typing import Dict
import requests
import zipfile
from pathlib import Path
import hashlib
import json
from os import system

CADASTRE_FOLDER_PATH="LaCie_thebaulm/gis/vectors/cadastre"
BASE_ETALAB_URL="https://cadastre.data.gouv.fr/data/etalab-cadastre" # etalab
BASE_RNB_URL="https://rnb-opendata.s3.fr-par.scw.cloud/files" # R.N.B. : Référentiel National des Bâtiments
CHUNCK_SIZE=8192

'''
Exemple:
Departement 09 Ariege
URL
    https://rnb-opendata.s3.fr-par.scw.cloud/files/RNB_09.csv.zip
URL stable
    https://www.data.gouv.fr/api/1/datasets/r/f2d0ecc2-f4c7-4bcd-be24-dce215e2df1a

A propos de l'URL dite stable
https://www.data.gouv.fr/api/1/datasets/r/<RESOURCE_ID>
<RESOURCE_ID> est un identifiant UUID d’une ressource précise (un fichier).
Il ne dépend pas directement du département.
'''

def die(msg):
    print(msg)
    sys.exit(1)


def main(data_type : str, dep_code : str, date = "yyyy-mm-dd"):
    
    # construction des repertoires de destination
    
    
    # nom du fichier de vecteurs raw layer
    if(data_type == "etalab"):
        raw_dirname = f"cadastre-{dep_code}-batiments-shp"
        etalab_param = build_etalab_param(dep_code, date)
        url = etalab_param["zip_url"]
        zip_filename = rf"{raw_dirname}.zip"
        output_filename = etalab_param["output_filename"]
        
    elif(data_type == "rnb"):
        raw_dirname = f"cadastre-{dep_code}-batiments-csv"
        rnb_param = build_rnb_param(dep_code)
        url = rnb_param["zip_url"]
        date = get_rnb_date(url)
        output_filename = rnb_param["output_filename"]
        zip_filename = rf"{output_filename}.zip"
        
        
    else:
        raise ValueError(f"Invalid data type: {args.data_type}. Expected 'etalab' or 'rnb'.")
    # end if
    
    home = Path.home()
    base_dir = home / CADASTRE_FOLDER_PATH

    # Check that LaCie is connected
    if not base_dir.exists():
        die(f"{base_dir} not found")

    rootp = base_dir / date
    zipped_dir = rootp / "zipped"
    unzipped_dir = rootp / "unzipped" / raw_dirname

    zipped_dir.mkdir(parents=True, exist_ok=True)
    unzipped_dir.mkdir(parents=True, exist_ok=True)
    
    zip_file_path = zipped_dir / zip_filename
    unzipped_file_path = unzipped_dir / output_filename
    
      # Check if data already exists
    if unzipped_file_path.exists():
        die(
            f"Data exists in {unzipped_dir}\n"
            "Use it or remove the directory"
    )
    
    sha1_of_zip_file = download_file(url, zip_file_path)
    
    print(f"Downloading is finished of {url} - sha1: {sha1_of_zip_file}")
    system(f"tree {rootp}")
    unzip(zip_file_path, unzipped_dir)

def download_file(url, output_path, chunk_size=CHUNCK_SIZE):
    """
    Télécharge fichier depuis une url vers un dossier cible et retourne sha1 pour verifier l'integrite du fichier téléchargé
    """
    
    sha1_of_file = hashlib.sha1()
    
    print(f"Downloading {url}")
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download file:\n{url}")

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                sha1_of_file.update(chunk)
    return sha1_of_file.hexdigest()

def unzip(zip_path, target_dir):
    """
    Dézippe une archive vers un dossier cible
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
            
    except Exception:
        die(
            f"Failed to extract content of:\n\t{zip_path}\n"
            f"in:\n\t{target_dir}"
        )
        
def get_csv_metadata(url : str) -> Dict:
    try:
        r = requests.head(url)
        r.raise_for_status()
        print(r.headers)
        metadata = {
            "content_length": r.headers.get("Content-Length"), 
            "last_modified": r.headers.get("Last-Modified"),
            "etag": r.headers.get("ETag"),
            "content_type": r.headers.get("Content-Type")
        }
        # print(json.dumps(metadata, indent=2))
        return metadata
    
    # gestion des erreurs
    except requests.exceptions.HTTPError as e:
        die(
            "HTTP error occurred"
            f"Status: {e.response.status_code}"
            f"Message: {e.response.text}"
        )
    except requests.exceptions.RequestException as e:
        die(f"Request failed {e}")

def build_rnb_param(dep_code) -> Dict:
    fname=rf"RNB_{dep_code}.csv"
    rnb_param = {
        "output_filename": fname,
        "zip_url": rf"{BASE_RNB_URL}/{fname}.zip"
    }
    return rnb_param  

def build_etalab_param(dep_code : str, date : str) -> Dict:
    etalab_param = {
        "output_filename": "batiments.shp",
        "zip_url": rf"{BASE_ETALAB_URL}/{date}/shp/departements/{dep_code}/cadastre-{dep_code}-batiments-shp.zip"
    }
    return etalab_param


def get_rnb_date(url: str):
    '''
    recupere la date de derniere modification
    '''
    metadata = get_csv_metadata(url)
    dt = datetime.strptime(metadata.get("last_modified"), "%a, %d %b %Y %H:%M:%S GMT")
    return dt.strftime("%Y-%m-%d")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Download and extract cadastre rnb-data as geojson for a department"
    )
    parser.add_argument("--data_type", type=str, required=True, help="Type of data: 'etalab' or 'rnb'") 
    parser.add_argument("--dep", type=str, required=True, help="Department number (XX or DOM-TOM)")
    args = parser.parse_args()
    data_type = args.data_type
    # Normalize department number
    dep_code = args.dep.strip()
    if len(dep_code) == 1:
        dep_code = f"0{dep_code}"
    elif len(dep_code) == 3 and dep_code.startswith("0"):
        dep_code = dep_code[1:]
     
    # Téléchargement depuis Etalab ou RNB ?
    if(data_type == "etalab"):
        parser.add_argument("--date", type=str, required=True, help='Date format must be: YYYY-MM-JJ.\nAvailable month ("MM") must be checked on: https://cadastre.data.gouv.fr/datasets/cadastre-etalab')
        args = parser.parse_args()
        date = args.date
    elif(args.data_type == "rnb"):
        pass
    else:
        raise ValueError(f"Invalid data type: {args.data_type}. Expected 'etalab' or 'rnb'.")
    # end if
    
    main(data_type, dep_code, date) 