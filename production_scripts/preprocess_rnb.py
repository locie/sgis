# exemple appel:
"""
Modify the raw cadastre layer of RNB by:

- adding a buffer distance around building (default 4m)
- adding a unique ID to each building
- removing small buildings (ground footprint < 10m2)

Produced data is stored locally.

example:

activate_PV_detection;
export QT_QPA_PLATFORM=offscreen;

"""

from re import match
from datetime import datetime
from pathlib import Path
import argparse
from production_scripts.target_folders_manager import TargetFoldersManager

# Add src folder to path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from sgis.vector_tools import VectorTools


def die(msg):
    print(msg)
    sys.exit(1)

# Fichiers en entree: les fichiers bruts du cadastre de RNB
# https://www.data.gouv.fr/datasets/referentiel-national-des-batiments

def main(dep):    
    target_folders_mng = TargetFoldersManager(dep,"2026"); 
    
    if len(dep)==2:
        prefix = f'0{dep}'
    else:
        prefix = dep
    
    qgis_vec_tools = VectorTools()
    # instantiate
    raw_vector = qgis_vec_tools.load_layer(vector_layer_path_raw, f'batiments_{dep}')
    raw_vector = qgis_vec_tools.copy_layer(raw_vector)
    preprocessed_vector, unwanted_buildings_number, initial_buildings_number = qgis_vec_tools.remove_small_features(raw_vector, min_area=10)
    preprocessed_vector = qgis_vec_tools.add_buffer_distance(preprocessed_vector, distance=4)
    preprocessed_vector = qgis_vec_tools.add_ID(preprocessed_vector, prefix=prefix)
    preprocessed_vector = qgis_vec_tools.add_XY_coordinates(preprocessed_vector)
    qgis_vec_tools.export_shp(preprocessed_vector, vector_layer_dir_preprocessed, name_preprocessed)

    with open(f'{output_root_path}/version_cadastre', 'w') as f:
        f.write(version_cadastre)
    with open(f'{output_root_path}/version_BDORTHO', 'w') as f:
        version_BDORTHO = f'{year}, BDOrtho database, IGN (RGB, resolution {resolution}cm)'
        f.write(version_BDORTHO)

    # vérification que les attributs sont les bons, par exemple code dep sur 3 digits
    attributes = preprocessed_vector.getFeature(500).attributeMap()
    if attributes['ID'][:3] != prefix:
        attr = attributes['ID'][:3]
        raise NameError(f'Bad prefix for images. Expected {prefix} got {attr}.')
    
    qgis_vec_tools.close()

if __name__ == "__main__":
   # allows code to run only when the script is executed, not when it’s imported !
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dep", type=str, required=True, help="Department code: 2 digits from 01 to 99 else 3 digits")
    args = parser.parse_args()
    
    main(args.dep)
