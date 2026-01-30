# cas d'absence de session X (i.e. pas de support Qt)
# solution: déclarer une variable d'env:
#       os.environ["QT_QPA_PLATFORM"] = "offscreen"
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from production_scripts import preprocess

# input params
dep='09';
year='2025';
cadastre_dir='2025-12-01'
resolution=20;

preprocess.main(dep, year, cadastre_dir, resolution)