import pandas as pd

from production_scripts.download_vectors import _count_non_polygon_geom

# Vérifie qu'il n'y a pas de doublons dans le fichier RNB
rnb_csv_file = "/home/thebaulm/LaCie_thebaulm/gis/vectors/cadastre/2026-04-04/unzipped/cadastre-31-batiments-csv/RNB_31.csv"
rnb = pd.read_csv(rnb_csv_file, usecols=["rnb_id"], dtype=str)

print("check from: ", rnb_csv_file)
# verif 1
print(len(rnb), "rows read from RNB CSV")
duplicates = rnb[rnb["rnb_id"].duplicated(keep=False)]
has_duplicates = rnb["rnb_id"].duplicated().any()
print(f"Has duplicates: {has_duplicates}")
print(duplicates)

# verif 2
print(f"Total rows: {len(rnb)} (expected : 736721)")
print(f"Unique rnb_id: {rnb['rnb_id'].nunique()}")
print(f"Duplicates: {rnb['rnb_id'].duplicated().sum()}")

not_polygon = _count_non_polygon_geom(rnb_csv_file)
print(f"Number of non-polygon geometries: {not_polygon}")

print("expected number:", len(rnb)-not_polygon)



# dans le dossier d'images
from pathlib import Path
from collections import Counter

folder = Path("/home/thebaulm/split/31/2025/rasters/images")
print("check from: ", folder)

# Récupérer les images
images = [
    f.name for f in folder.iterdir()
    if f.is_file() and f.suffix.lower() in {".jpg"}
]

# Partie avant "_"
prefixes = [
    image.split("_", 1)[0]
    for image in images
]

# Comptage
counts = Counter(prefixes)

# Doublons uniquement
duplicates = {name: count for name, count in counts.items() if count > 1}

# for name, count in sorted(duplicates.items()):
#     print(f"{name}: {count}")
    
print(f"Nombre total d'images : {len(images)}")
print(f"Nombre de préfixes uniques : {len(counts)}")
print(f"Nombre de doublons : {sum(count - 1 for count in duplicates.values())}")






from collections import Counter

txt_file = "/home/thebaulm/split/31/2025/rasters/progress.txt"
print('check progress file: ', txt_file)

with open(txt_file, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

counts = Counter(lines)

duplicates = {line: count for line, count in counts.items() if count > 1}

print(f"Nombre de lignes : {len(lines)}")
print(f"Nombre de lignes uniques : {len(counts)}")
print(f"Nombre de doublons : {sum(count - 1 for count in duplicates.values())}")

for line, count in duplicates.items():
    print(f"{count}x : {line}")
    
    
from pathlib import Path
import shutil



progress_file = Path("/home/thebaulm/split/31/2025/rasters/progress.txt")
csv_dir = Path("/home/thebaulm/split/31/2025/rasters/repartition_by_rasters")

# shutil.copy2(progress_file, progress_file.with_suffix(".txt.bak"))

# CSV disponibles
csv_names = {f.stem for f in csv_dir.glob("*.csv")}

# Lire progress.txt
with progress_file.open("r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# Conserver uniquement les lignes ayant un CSV correspondant
valid_lines = [
    line for line in lines
    if Path(line).stem in csv_names
]

missing = len(lines) - len(valid_lines)

# Réécrire progress.txt
# with progress_file.open("w", encoding="utf-8") as f:
#     for line in valid_lines:
#         f.write(line + "\n")

print(f"Lignes initiales : {len(lines)}")
print(f"Lignes supprimées : {missing}")
print(f"Lignes restantes : {len(valid_lines)}")


