#!/usr/bin/env python3
from datetime import datetime
from itertools import groupby
from os import environ
from pathlib import Path
import re

BASE_DIR="LaCie_thebaulm" # Nom du dossier de stockage des donnees d'entree et de sortie à la racine du home de l'utilisateur.

def scan_cadastre_vectors_folder(data_type : str):
    """
    Scanne les dossiers de cadastre et affiche les chemins des dossiers les plus récents pour chaque numéro de cadastre, triés par date croissante.
    """
    if(data_type == "etalab"):
        pattern_str = r"cadastre-(\d+)-batiments-shp"
    elif(data_type == "rnb"):
        pattern_str = r"cadastre-(\d+)-batiments-csv"
    else:
        raise ValueError(f"Invalid data type: {data_type}. Expected 'etalab' or 'rnb'.")

    
    home_path = environ['HOME']
    path_to_scan = rf'{home_path}/{BASE_DIR}/gis/vectors/cadastre'
    root_path = Path(path_to_scan)
    cadastre_pattern = re.compile(pattern_str)

    # cherche tous les dossiers qui matchent le pattern "cadastre-{number}-batiments-shp"
    matching_dirs = [p for p in root_path.rglob("*") if p.is_dir() and cadastre_pattern.fullmatch(p.name)]
    

    # # Keep only paths where the last folder is not empty
    # matching_dirs = [
    #     p for p in matching_dirs
    #     if any(p.iterdir())  
    # ]
            
    # cadastre ascending
    matching_dirs_sorted = sorted(
        matching_dirs,
        key=lambda p:(
            int(re.search(pattern_str, p.name).group(1)),  
            # -datetime.strptime(p.parts[7], "%Y-%m-%d").timestamp() 
        )
    )
    
    # pour chaque numéro de cadastre, garde le dossier le plus récent (en se basant sur la date dans le chemin)
    most_recent_by_cadastre_number = [
        max(group, key=lambda p: datetime.strptime(p.parts[7], "%Y-%m-%d"))
        for cadastre, group in groupby(
            matching_dirs_sorted,
            key=lambda p:  int(re.search(pattern_str, p.parts[9]).group(1))
        )
    ]
    
    # date ascending
    final_paths = sorted(
        most_recent_by_cadastre_number,
        key=lambda p:(
            # int(re.search(r"cadastre-(\d+)-batiments-shp", p.name).group(1)),  
            datetime.strptime(p.parts[7], "%Y-%m-%d").timestamp() 
        )
    )
    
    # Group paths by date
    print("\n------------------------------------------------------------------")
    print("chemins des dossiers les plus récents pour chaque numéro de cadastre, triés par date croissante :")
    print("------------------------------------------------------------------")
    date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
    for date, group in groupby(
        final_paths,
        key=lambda p: next(datetime.strptime(part, "%Y-%m-%d")
                        for part in p.parts if date_pattern.fullmatch(part))
    ):
        print(f"Date {date.date()}:")
        for p in group:
            print(f"  {p}")
        print()
    