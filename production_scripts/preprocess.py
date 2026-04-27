# exemple appel:
"""
Modify the raw layer from Etalab or RNB data:

- adding a buffer distance around building (default 4m)
- adding a unique ID to each building
- removing small buildings (ground footprint < 10m2)

Produced data is stored locally.

RNB usage :

activate_PV_detection;
export PYTHONPATH=~/sgis:~/sgis/src:~/sgis/production_scripts:$PYTHONPATH
data_type=rnb;dep=09;year=2025;resolution=20;cadastre_dir=2026-03-28
python ~/sgis/production_scripts/preprocess.py --data_type=$data_type --dep $dep --year=${year} --cadastre_dir ${cadastre_dir} --resolution $resolution

Etalab usage :

activate_PV_detection;
export PYTHONPATH=~/sgis:~/sgis/src:~/sgis/production_scripts:$PYTHONPATH
data_type=etalab;dep=09;year=2025;resolution=20;cadastre_dir=2025-12-01
python ~/sgis/production_scripts/preprocess.py --data_type=$data_type --dep $dep --year=${year} --cadastre_dir ${cadastre_dir} --resolution $resolution

"""

import argparse
from production_scripts._preprocess_etalab import PreprocessEtalab
from production_scripts._preprocess_rnb import PreprocessRNB

def main(data_type : str, dep : str, cadastre_dir : str, BDortho_raster_year : str, resolution = 20, raw_folder_path = None, dest_folder_path = None):
     
      match (data_type):
            case ("etalab"):
                  # ETALAB
                  etalab_preprocess_instance = PreprocessEtalab(dep, cadastre_dir, BDortho_raster_year, resolution, raw_folder_path, dest_folder_path)
                  etalab_preprocess_instance.run()   
            case ("rnb"):
                  # RNB
                  rnb_preprocess_instance = PreprocessRNB(dep, cadastre_dir, BDortho_raster_year, resolution, raw_folder_path, dest_folder_path)
                  rnb_preprocess_instance.run()   
            case _:
                  raise ValueError(f"Invalid data type: {args.data_type}. Expected 'etalab' or 'rnb'.")
       
           
if __name__ == "__main__":
      # parse command line arguments
      parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
      parser.add_argument("--data_type", type=str, required=True, help="Type of data: 'etalab' or 'rnb'")
      parser.add_argument("--dep", type=str, required=True, help="Department code: 2 digits from 01 to 99 else 3 digits")
      parser.add_argument("--year", type=str, required=True, help='BDOrtho version [YYYY]')
      parser.add_argument("--cadastre_dir", required=True, 
                              help="The name of the corresponding data diretory on ~/LaCie_thebaulm. [YYYY-MM-DD]" 
                              "A similar name is exported in file `version_cadastre`.")
      parser.add_argument("--resolution", type=int, default=20, help='Raster resolution, in cm. Exported in `version_BDORTHO`.')
      args = parser.parse_args()
     
      # launch main function with parsed arguments
      main(args.data_type, args.dep, args.cadastre_dir, args.year, args.resolution)

      