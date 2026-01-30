# exemple appel:
"""
Modify the raw cadastre layer of Etalab by:

- adding a buffer distance around building (default 4m)
- adding a unique ID to each building
- removing small buildings (ground footprint < 10m2)

Produced data is stored locally.

example:

activate_PV_detection;export QT_QPA_PLATFORM=offscreen;
dep=10;year=2024;resolution=20;cadastre_dir=2026-01-01
python ~/sgis/production_scripts/preprocess.py --dep $dep --year $year --cadastre_dir ${cadastre_dir} --resolution $resolution > ~/split/notes_$dep_$year.temp;
cat ~/sgis/production_scripts/notes_template.txt >> ~/split/notes_$dep_$year.temp; 
mv ~/split/notes_$dep_$year.temp ~/split/$dep/$year/notes_${dep}
"""
# fixme: pb en cas d'absence de session X (i.e. pas de support Qt)
# solution: déclarer une variable d'env:
#       os.environ["QT_QPA_PLATFORM"] = "offscreen"
# non testé sur l'ensemble de sgis

# from sgis.splitter import Splitter, clean_processing_folder

from re import match
from datetime import datetime
from sgis.vector_tools import *
from pathlib import Path
from os import environ

# stuff to run always here such as class/def
def main(dep, year, cadastre_dir, resolution=20):    
    
    #  checks input parameters before preprocessing
    assert ((len(dep)==2) or (len(dep)==3 and dep[0]=='9'))
    assert resolution in range(1, 100)
    try:
        date = datetime.strptime(cadastre_dir, '%Y-%m-%d')
    except ValueError as e:
        print("\n\n Invalid date format. Expected YYYY-MM-DD:")
        raise e

    version_cadastre = datetime.strftime(date, '%B, %Y, Cadastre Etalab')

    # création des dossiers (fixme unsafe for Windows)
    home = environ['HOME']       # requis pour s'adapter à Kheops ET à Cleo
    vector_layer_path_raw =          rf'{home}/LaCie_thebaulm/gis/vectors/cadastre/{cadastre_dir}/unzipped/cadastre-{dep}-batiments-shp/batiments.shp'
    name_preprocessed =                'batiments.shp'
    raster_layers_path =             rf'{home}/temporary_LaCie/rasters/only_tiles/{dep}/{year}/{dep}-{year}-0M{resolution}-RGB'
    output_root_path =               rf'{home}/split/{dep}/{year}'

    output_directory_path =          rf'{output_root_path}/rasters'
    vector_layer_dir_preprocessed =  rf'{output_root_path}/preprocessing/vectors'             
    vector_layer_path_preprocessed = rf'{vector_layer_dir_preprocessed}/{name_preprocessed}'  


    for p in (vector_layer_dir_preprocessed, output_directory_path):#
        try:
            Path(p).mkdir(parents=True) # `exist_ok=True` is useless here because p is a directory
        except FileExistsError as e:
            raise e

    if not Path(vector_layer_path_raw).exists():
        raise FileNotFoundError(f'Cadastre data not found in '
                                f'{home}/LaCie_thebaulm/gis/vectors/cadastre/{cadastre_dir}/unzipped/cadastre-{dep}-batiments-shp')

    if len(dep)==2:
        prefix = f'0{dep}'
    else:
        prefix = dep
        
    raw_vector = load_layer(vector_layer_path_raw, f'batiments_{dep}')
    raw_vector = copy_layer(raw_vector)
    preprocessed_vector, unwanted_buildings_number, initial_buildings_number = remove_small_features(raw_vector, min_area=10)
    preprocessed_vector = add_buffer_distance(preprocessed_vector, distance=4)
    preprocessed_vector = add_ID(preprocessed_vector, prefix=prefix)
    preprocessed_vector = add_XY_coordinates(preprocessed_vector)
    export_shp(preprocessed_vector, vector_layer_dir_preprocessed, name_preprocessed)

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

if __name__ == "__main__":
   # allows code to run only when the script is executed, not when it’s imported !
   
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dep", type=str, required=True, help="Department code: 2 digits from 01 to 99 else 3 digits")
    parser.add_argument("--year", type=str, required=True, help='BDOrtho version [YYYY]')
    parser.add_argument("--cadastre_dir", required=True, 
                        help="Etalab version, and the name of the corresponding data diretory on ~/LaCie_thebaulm. [YYYY-MM-DD]" 
                        "A similar name is exported in file `version_cadastre`.")
    parser.add_argument("--resolution", type=int, default=20, help='Raster resolution, in cm. Exported in `version_BDORTHO`.')
    args = parser.parse_args()
    
    main(args)
