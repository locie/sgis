#!/usr/bin/env python3

import argparse
import os
import sys
import requests
import zipfile
from pathlib import Path
import hashlib
import json

CADASTRE_FOLDER_PATH="LaCie_thebaulm/gis/vectors/cadastre"
BASE_RNB_URL="https://rnb-opendata.s3.fr-par.scw.cloud/files/" #R.N.B. : Référentiel National des Bâtiments

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


def main(date, dep_code):
    home = Path.home()
    base_dir = home / CADASTRE_FOLDER_PATH

    # Check that LaCie is connected
    if not base_dir.exists():
        die(f"{base_dir} not found")

    print(f"Department: {dep_code}")

    rootp = base_dir / date
    filename = f"cadastre-{dep_code}-batiments-shp"

    zipped_dir = rootp / "zipped"
    unzipped_dir = rootp / "unzipped"
    target_unzipped = unzipped_dir / filename

    # Check if data already exists
    if target_unzipped.exists():
        die(
            f"Data exists in {target_unzipped}\n"
            "Use it or remove the directory"
        )

    zipped_dir.mkdir(parents=True, exist_ok=True)
    unzipped_dir.mkdir(parents=True, exist_ok=True)

    url = (
        BASE_RNB_URL /
        f"RNB_{dep_code}.csv.zip"
    )
    zip_path = zipped_dir / f"{filename}.zip"
    
    download_file(url, zip_path)
    unzip(zip_path, unzipped_dir)
            
    die(f"Download and extraction completed successfully {url}")

def download_file(url, output_path, chunk_size=8192):
    """Telecharge fichier depuis une url vers un dossier cible et retourne sha1 pour verifier l'integrite du fichier telecharge"""
    
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
    """Dezippe une archive vers un dossier cible"""
    
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
            
    except Exception:
        die(
            f"Failed to extract content of:\n\t{zip_path}\n"
            f"in:\n\t{target_dir}"
        )
        
def get_csv_metadata(url):
    try:
        r = requests.head(url)
        r.raise_for_status()

        metadata = {
            "content_length": r.headers.get("Content-Length"), 
            "last_modified": r.headers.get("Last-Modified"),
            "etag": r.headers.get("ETag"),
            "content_type": r.headers.get("Content-Type")
        }
        print(json.dumps(metadata, indent=2))
    
    # gestion des erreurs
    except requests.exceptions.HTTPError as e:
        die(
            "HTTP error occurred"
            f"Status: {e.response.status_code}"
            f"Message: {e.response.text}"
        )
    except requests.exceptions.RequestException as e:
        die(f"Request failed {e}")


if __name__ == "__main__":
    # allows code to run only when the script is executed, not when it’s imported !
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Download and extract cadastre rnb-data as geojson for a department"
    )
    parser.add_argument("--date", type=str, required=True, help="Date (YYYY-MM-JJ)")
    parser.add_argument("--dep", type=str, required=True, help="Department number (XX or DOM-TOM)")
    args = parser.parse_args()
    
    date = args.date
    dep_code = args.dep.strip()

    # Normalize department number
    if len(dep_code) == 1:
        dep_code = f"0{dep_code}"
    elif len(dep_code) == 3 and dep_code.startswith("0"):
        dep_code = dep_code[1:]
    
    main(date, dep_code) 