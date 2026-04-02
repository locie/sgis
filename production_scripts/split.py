"""
Split the raster files of a department (BDORtho from IGN) according to the preprocessed version of the cadastre (Etalab).
In case the splitting process is interrupted, it will restart from the last processed raster file.

RNB usage:

activate_PV_detection;
export PYTHONPATH=~/sgis:~/sgis/src:~/sgis/production_scripts:$PYTHONPATH
dep=09;year=2026;resolution=20;threads_num=6
python ~/sgis/production_scripts/split.py --dep $dep --year $year --resolution $resolution --threads_num ${threads_num}

Etalab usage :

activate_PV_detection;
export PYTHONPATH=~/sgis:~/sgis/src:~/sgis/production_scripts:$PYTHONPATH
dep=09;year=2025;resolution=20;threads_num=6
python ~/sgis/production_scripts/split.py --dep $dep --year $year --resolution $resolution --threads_num ${threads_num}


Known bug: splitting process may fail whenever few rasters are remaining and threads_num > 1.
Typical splitting speed is 3-5 days per department, using one thread.
"""
import argparse
from _production_constants import *
from sgis.splitter import Splitter
from os import environ
from pathlib import Path

def main(dep : str, year : str, resolution : int, threads_num = int, raw_folder_path = None, dest_folder_path = None):
    
    #  construit les chemins d'entree/sortie
    if(raw_folder_path == None):
        home_path = Path(environ['HOME'])
        raw_folder_path = home_path / RAW_FOLDERNAME
        dest_folder_path = home_path
    else:
        raw_folder_path = Path(raw_folder_path)
        dest_folder_path =  Path(dest_folder_path)
        
    name_preprocessed = PREPROCESSED_BUILDINGS_LAYER_FILENAME
    raster_layers_path = dest_folder_path / RELATIVE_TEMP_TILES_FOLDER_PATH / dep / year / rf"{dep}-{year}-0M{resolution}-RGB"
    output_root_path = dest_folder_path / "split" / dep / year

    output_directory_path = output_root_path / "rasters"
    vector_layer_dir_preprocessed_path =  output_root_path / "preprocessing" / "vectors"             
    vector_layer_path_preprocessed_path = vector_layer_dir_preprocessed_path / name_preprocessed 
        
    # GO !
    splitter = Splitter(vector_layer_path_preprocessed_path, raster_layers_path, output_directory_path)
    splitter.split(threads_num=threads_num) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--year", type=str, required=True, help='BDOrtho version [YYYY]')
    parser.add_argument("--dep", type=str, required=True, help="Department code: 2 digits from 01 to 99 else 3 digits")
    parser.add_argument("--threads_num", type=int, default=1, choices=range(1, 7), help='Number of concurrent threads for splitting. Each thread splits one raster tile.')
    parser.add_argument("--resolution", type=int, default=20, help='Raster resolution, in cm.')
    args = parser.parse_args()

    dep = args.dep
    year = args.year
    threads_num = args.threads_num
    resolution = args.resolution
    assert ((len(dep)==2) or (len(dep)==3 and dep[0]=='9'))
    main(dep, year, resolution, threads_num)
