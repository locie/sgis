"""
Split the raster files of a department (BDORtho from IGN) according to the preprocessed version of the cadastre (Etalab).
In case the splitting process is interrupted, it will restart from the last processed raster file.

example:
activate_PV_detection;export QT_QPA_PLATFORM=offscreen;
dep=75;year=2011;resolution=20;threads_num=1
python ~/sgis/production_scripts/split.py --dep $dep --year $year --resolution $resolution --threads_num ${threads_num} 2>~/split/$dep/$year/split.log

Known bug: splitting process may fail whenever few rasters are remaining and threads_num > 1.
Typical splitting speed is 3-5 days per department, using one thread.
"""

import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--year", type=str, required=True, help='BDOrtho version [YYYY]')
parser.add_argument("--dep", type=str, required=True, help="Department code: 2 digits from 01 to 99 else 3 digits")
parser.add_argument("--threads_num", type=int, default=1, choices=range(1, 6), help='Number of concurrent threads for splitting. Each thread splits one raster tile.')
parser.add_argument("--resolution", type=int, default=20, help='Raster resolution, in cm.')

args = parser.parse_args()


dep = args.dep
year = args.year
threads_num = args.threads_num
resolution = args.resolution


 
assert ((len(dep)==2) or (len(dep)==3 and dep[0]=='9'))


from sgis.splitter import Splitter, clean_processing_folder
from pathlib import Path
from os import environ




# Création des dossiers (fixme unsafe for Windows)

home = environ['HOME']       # requis pour s'adapter à Kheops ET à Cleo
name_preprocessed =                'batiments.shp'
raster_layers_path =             rf'{home}/temporary_LaCie/rasters/only_tiles/{dep}/{year}/{dep}-{year}-0M{resolution}-RGB'
output_root_path =               rf'{home}/split/{dep}/{year}'

    
output_directory_path =          rf'{output_root_path}/rasters'
vector_layer_dir_preprocessed =  rf'{output_root_path}/preprocessing/vectors'             
vector_layer_path_preprocessed = rf'{vector_layer_dir_preprocessed}/{name_preprocessed}'  




splitter = Splitter(vector_layer_path_preprocessed, raster_layers_path, output_directory_path)
splitter.split(threads_num=threads_num)