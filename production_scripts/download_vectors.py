#!/usr/bin/env python3
import argparse
import sys
import requests
import zipfile
from pathlib import Path
import hashlib
import pandas as pd
from production_scripts.cadastre_data import RNB, Etalab
from production_scripts.rnb_geo_api import normalize_dept_code_number
from production_scripts.rnb_geo_api import request_all_rnb_csv_metadata

RAW_DATA_BASE_FOLDER="LaCie_thebaulm"
RELATIVE_VECTORS_CADASTRE_FOLDER_PATH=rf"gis/vectors/cadastre"
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

def main(dept_code : str, data_type="rnb", date = "yyyy-mm-dd", raw_folder_path = None):
    #  construit les chemins d'entree/sortie
    if(raw_folder_path == None):
        home_path = Path.home()
        raw_folder_path = home_path / RAW_DATA_BASE_FOLDER
        if not raw_folder_path.exists():
            die(f"{raw_folder_path} not found")
    else:
        raw_folder_path = Path(raw_folder_path)
    
    cadastre_dir = raw_folder_path / RELATIVE_VECTORS_CADASTRE_FOLDER_PATH
    if data_type == "etalab":
        data = Etalab.from_dep(dept_code, date)

    elif data_type == "rnb":
        data = RNB.from_dep(dept_code)

    else:
        raise ValueError(f"Invalid data type: {data_type}. Expected 'etalab' or 'rnb'.")

    zip_file_path, unzipped_file_path = data.build_paths(cadastre_dir)

    if unzipped_file_path.exists():
        die(f"Data exists in {unzipped_file_path.parent}\nUse it or remove the directory")
    
    # download raw file
    f_hash = download_file(data.url, zip_file_path)
    
    # verify sha1 before unzip (only for RNB because sha1 is unavailable for etalab)
    if(data_type == "rnb"):
        if f_hash != data.expected_hash:
            raise ValueError(f"Downloading is finished of {data.url} but actual sha1: {f_hash} mismatchs the expected sha1: {data.expected_hash} ")
        
    
    # unzip raw data 
    unzip(zip_file_path, unzipped_file_path.parent)
    
    if(data_type == "rnb"):
        # NOTE ⚠ IMPORTANT : Nettoie la colonne "shape" avant traitement
        # Suppression de 'SRID=4326;' dans la colonne "shape" WKT Multipolygone
        _fix_shape_column_for_qgis(unzipped_file_path)
        not_polygon = _count_non_polygon_geom(unzipped_file_path)
        print(f"{unzipped_file_path} contains {not_polygon} geometries that neither POLYGON nor MULTIPOLYGON.")
    else:
        print(f"{unzipped_file_path}")
    return data

def download_all_depts():
    all_available_rnb_metadata = request_all_rnb_csv_metadata()
    for c in all_available_rnb_metadata:
        main(c.dept_code, data_type="rnb")
    
def download_file(url, output_path, chunk_size=CHUNCK_SIZE, expected_sha1=None):
    """
    Télécharge fichier depuis une url vers un dossier cible et retourne sha1 pour verifier l'integrite du fichier téléchargé
    """
    
    hash = hashlib.sha1()
    
    print(f"Downloading {url}")
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download file:\n{url}")

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                hash.update(chunk)
    return hash.hexdigest()


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
        
# ============================================================
# ⚠ IMPORTANT : Nettoyer le champ WKT avant traitement
# ============================================================
def _fix_shape_column_for_qgis(csv_path : Path):
    '''
    Cette fonction modifie le fichier CSV d’entrée directement en supprimant les déclarations SRID=4326
    des chaînes de géométrie WKT (Well-Known Text). Les préfixes SRID peuvent provoquer des problèmes
    lors de l’importation des géométries dans QGIS ; cette étape de prétraitement garantit donc la compatibilité.
    '''
    df = pd.read_csv(csv_path, sep=';')
    df['shape'] = df['shape'].str.replace(r'^SRID=4326;', '', regex=True)
    df.to_csv(csv_path, index=False)


def _count_non_polygon_geom(csv_path : str) -> int:
    df = pd.read_csv(csv_path, sep=',')
    # Count geometries that not mismatchs POLYGON or MULTIPOLYGON
    pattern = r"^(?:POLYGON|MULTIPOLYGON)"
    notpolygon_counts = (~df["shape"].str.contains(pattern, na=False)).sum()
    # print("Nombre de lignes ne contenant PAS POLYGON ou MULTIPOLYGON :", notpolygon_counts)
    return notpolygon_counts


def download_all_available_rnb_csv(list_to_dwld : list):
    for e in list_to_dwld:
        main(data_type, dept_code)
    # system(f"tree {rootp}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Download and extract cadastre rnb-data as geojson for a department"
    )
    parser.add_argument("--data_type", type=str, required=True, help="Type of data: 'etalab' or 'rnb'") 
    parser.add_argument("--dep", type=str, required=True, help="Department number (XX or DOM-TOM)")
    args = parser.parse_args()
    data_type = args.data_type
    
    dept_code = normalize_dept_code_number(args.dep)
     
    # Téléchargement depuis Etalab ou RNB ?
    if(data_type == "etalab"):
        parser.add_argument("--date", type=str, required=True, help='Date format must be: YYYY-MM-JJ.\nAvailable month ("MM") must be checked on: https://cadastre.data.gouv.fr/datasets/cadastre-etalab')
        args = parser.parse_args()
        date = args.date
        main(dept_code, data_type, date) 
    elif(args.data_type == "rnb"):
        main(dept_code) 
    else:
        raise ValueError(f"Invalid data type: {args.data_type}. Expected 'etalab' or 'rnb'.")
    # end if
    
    