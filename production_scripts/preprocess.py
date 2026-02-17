import argparse
from unittest import case
from production_scripts._preprocess_etalab import PreprocessEtalab
# s_preprocess_rnb

def main(data_type, dep, year, cadastre_dir, resolution):
     
      match (data_type):
            case ("etalab"):
                  # ETALAB
                  etalab_preprocess_instance = PreprocessEtalab(dep, year, cadastre_dir, resolution)
                  etalab_preprocess_instance.run()   
            case ("rnb"):
                  # RNB
                  # TODO instantiate and run RNB preprocess
                  pass   
            case _:
                  raise ValueError(f"Invalid data type: {args.data_type}. Expected 'etalab' or 'rnb'.")
       
           
if __name__ == "__main__":
      # parse command line arguments
      parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
      parser.add_argument("--data_type", type=str, required=True, help="Type of data: 'etalab' or 'rnb'")
      parser.add_argument("--dep", type=str, required=True, help="Department code: 2 digits from 01 to 99 else 3 digits")
      parser.add_argument("--year", type=str, required=True, help='BDOrtho version [YYYY]')
      parser.add_argument("--cadastre_dir", required=True, 
                              help="Etalab version, and the name of the corresponding data diretory on ~/LaCie_thebaulm. [YYYY-MM-DD]" 
                              "A similar name is exported in file `version_cadastre`.")
      parser.add_argument("--resolution", type=int, default=20, help='Raster resolution, in cm. Exported in `version_BDORTHO`.')
      args = parser.parse_args()
     
      # launch main function with parsed arguments
      main(args.data_type, args.dep, args.year, args.cadastre_dir, args.resolution)

      