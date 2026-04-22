import argparse
import requests
import hashlib
from tqdm import tqdm
from pathlib import Path
from shutil import move
# import py7zr
import subprocess
from production_scripts.rnb_geo_api import normalize_dept_code_number

DOWNLOAD =    Path.home() / Path("temporary_LaCie/rasters/downloads.temp")
TILES_LACIE = Path.home() / Path("temporary_LaCie/rasters/only_tiles")
TILES =       Path.home() / Path("LaCie_thebaulm/gis/rasters/only_tiles")
ARCHIVES_LACIE =    Path.home() / Path("LaCie_thebaulm/gis/rasters/archives")
CHUNK_SIZE = 32768

class DownloadRaster():
    def __init__(self, dept_code, year, resolution, retry_download):
        self._dept_code = dept_code
        self._year = year
        self._resolution = resolution
        self._retry_download = retry_download
        self.check_existing_paths()
        possible = self.get_download_links()
        self.download(possible)
        self.extract(possible)
        self.move_files(dept_code)


    def check_existing_paths(self):
        for path in (TILES_LACIE, TILES, ARCHIVES_LACIE):
            if not path.exists():
                raise FileNotFoundError(path)
            path_ = path / self._dept_code / self._year
            if path_.exists():
                raise FileExistsError(path_)



    def get_download_links(self):
        print("Looking for download links:")
        dept_code = f"{self._dept_code:>03}"   # ajout d'un préfixe 0 si code dep sur 2 chiffres
        BASE = "https://data.geopf.fr/telechargement/download/BDORTHO"
        possible = {}
        if dept_code.startswith("9"): # DOM TOM
            CRS = ("LAMB93",
                        "RGAF09UTM20",
                        "RGFG95UTM22",
                        "RGM04UTM38S",
                        "RGR92UTM40S",
                        "RGR92UTM40S",
                        "RGSPM06U21",
                        "UTM20W84GUAD"
                        )
        else:
            CRS = ("LAMB93",)
        for path_component in tqdm(("R_1", "R_2", "O_1", "O_2")):
            for CRS_ in CRS:
                NAME = f"BDORTH{path_component}-0_RVB-0M{self._resolution}_JP2-E080_{CRS_}_D{dept_code}_{self._year}-01-01"

                # département en une seul archive
                NAME_PART = f"{NAME}.7z"
                URL = f"{BASE}/{NAME}/{NAME_PART}"
                response = requests.get(URL, stream=True)
                if response.status_code == 200:
                    possible[(path_component, CRS_)] = [(URL, NAME_PART)]
                    self._several_parts = False
                    continue


                # département en plusieurs sous-archives
                k = 1
                NAME_PART = f"{NAME}.7z.0{k:>02}"
                URL = f"{BASE}/{NAME}/{NAME_PART}"
                response = requests.get(URL, stream=True)

                while response.status_code == 200:
                    if k == 1:
                        possible[(path_component, CRS_)] = []
                    possible[(path_component, CRS_)].append((URL, NAME_PART))
                    k += 1
                    NAME_PART = f"{NAME}.7z.0{k:>02}"
                    URL = f"{BASE}/{NAME}/{NAME_PART}"
                    response = requests.get(URL, stream=True)
                self._several_parts = True

        if len(possible) == 0:
            raise ValueError("\nNo file found for these options, please visit the following link for direct download: " \
            "\n   https://cartes.gouv.fr/rechercher-une-donnee/dataset/IGNF_BD-ORTHO.")
        elif len(possible) > 1:
            raise ValueError("\nDetermination of files to be downloaded is ambiguous, please visit the following link for direct download: " \
            "\n   https://cartes.gouv.fr/rechercher-une-donnee/dataset/IGNF_BD-ORTHO.")
        else:
            possible = list(possible.values())[-1]
        return possible


    def download(self, possible):
        print("Downloading:")
        path_ = DOWNLOAD / self._dept_code / self._year
        try:
            path_.mkdir(parents=True, exist_ok=self._retry_download)
        except FileExistsError as e:
            raise FileExistsError(f"Directory {path_} exists. Pass '--retry_download' to skip this check.") from e
        
        # suivi progression via tqdm
        #   source: https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
        for URL, name_part in possible:
            if (path_ / name_part).exists():
                print(f"Skipping part {name_part} as it was found in {path_}")
            else:
                print("\t", name_part)
                response = requests.get(URL, stream=True)
                total_size = response.headers.get('content-length', 0)
                total_size = int(total_size)
                with open(path_ / name_part, "wb") as f:
                    with tqdm(name_part, total=total_size, unit="iB", unit_scale=True, unit_divisor=1024) as bar:
                        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                            if chunk is not None:
                                size = f.write(chunk)
                                bar.update(size)
        # verification md5
        URL_md5 =       possible[0][0].replace(".7z.001", ".md5").replace(".7z", ".md5")
        name_part_md5 = possible[0][1].replace(".7z.001", ".md5").replace(".7z", ".md5")
        response = requests.get(URL_md5, stream=True)
        if response.status_code != 200:
            print("No MD5 file found.")
        else:
            print("MD5 file found, checking content:")

            # téléchargement fichier signatures
            with open(path_ / name_part_md5, "wb") as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk is not None:
                        f.write(chunk)
            
            # lecture/interprétation fichier signatures
            with open(path_ / name_part_md5, "r") as f:
                lines = f.readlines()
        
            # parcours fichiers 7z et vérification
            for line in tqdm(lines):
                name_part       = line.split("/")[1][:-1]
                reference_md5   = line.split(" ")[0]

                with open(path_ / name_part, "rb") as f:
                    file_hash = hashlib.md5()
                    while chunk := f.read(CHUNK_SIZE):
                        file_hash.update(chunk)
                computed_md5 = file_hash.hexdigest()
                if computed_md5 != reference_md5:
                    print("\n\n")
                    print("Reference md5: ", reference_md5)
                    print("Computed md5:  ", computed_md5)
                    raise ValueError("Content of downloaded file does not match given md5 signature.")

    def extract(self, possible):
        """
        ## A propos de py7zr
        - py7zr est un package qui permet de manipuler des fichiers 7z via Python
        - l'extraction d'une archive multi-parties par py7zr requiert la concatenation des parties (*.7z.00*) en un fichier unique 
                source: https://py7zr.readthedocs.io/en/latest/user_guide.html#extraction-from-multi-volume-archive
        - cela cause un problème 
                - d'espace disque (doublon temporaire)
                - de lenteur supposé comparé à la commande 7z shell qui opère directement sur les fichiers téléchargés
        - pour ces raisons py7zr est remplacé par un appel à subprocess.run
        """
        ## avec py7zr
        # if self._several_parts: 
        #     print("Creating a single archive from all parts")
        #     concat_archive = "all_parts.7z"
        #     path_ = DOWNLOAD / dept_code / self._year / concat_archive
        #     with open(path_, "ab") as outfile:     # trop lent. Passer par subprocess
        #         for URL, name_part in possible:
        #             with open(path_ / name_part, "rb") as infile:
        #                 outfile.write(infile.read())
        # else:
        #     first_part = possible[0]
        #     URL, name_part = first_part
        #     path_ = DOWNLOAD / dept_code / self._year / name_part
        
        # print("Extracting")
        # try:
        #     archive = py7zr.SevenZipFile(path_, mode='r')
        # except Exception as e: # e.g.:  py7zr.exceptions.Bad7zFile
        #     raise FileNotFoundError(f"Something went wrong during download, delete your files in {path_.parent} and try again.") \
        #         from e
        # else:
        #     archive.extractall(path=path_.parent)
        #     archive.close()

        # Pour avoir des informations de progression lors de la décompression, utiliser subprocess.Popen
        first_part = possible[0]
        URL, name_part = first_part
        cmd = ["7z", "x", name_part]
        path_ = DOWNLOAD / dept_code / self._year
        with subprocess.Popen(cmd, cwd=path_, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
            for line in proc.stdout:
                print(line, end="")

    def move_files(self, dept_code):
        # détermination du dossier du contenu extrait
        path_ = DOWNLOAD / dept_code / self._year
        paths = [p for p in path_.iterdir() if p.is_dir()]
        if len(paths) != 1:
            raise FileNotFoundError(f"Something went wrong during download or extraction, check your files in {path_} and try again.")
        path_ = paths[0]

        print("Moving tiles")
        path_dest = TILES_LACIE / dept_code / self._year / f"{dept_code}-{self._year}-0M{self._resolution}-RGB"
        path_dest.mkdir(parents=True)
        for pattern in ("**/*.jp2", 
                        "**/*.tab", 
                        f"{dept_code}-{self._year}*.jp2.aux.xml"):
            for file in path_.glob(pattern):
                move(file, path_dest)

        print("Moving remaining to archive")
        path_dest = ARCHIVES_LACIE / dept_code / self._year
        path_dest.mkdir(parents=True)
        move(path_, path_dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Download and extract BDOrtho data for a department. Data is stored at https://cartes.gouv.fr/rechercher-une-donnee/dataset/IGNF_BD-ORTHO"
    )
    parser.add_argument("--dep", type=str, required=True, help="Department number (XX or DOM-TOM)")
    parser.add_argument("--year", type=str, required=True, help="Year must be: YYYY")
    parser.add_argument("--resolution", type=int, required=False, default=20, help="For instance, '20' for '20 cm/pixels'")
    parser.add_argument('--retry_download', default=False, action=argparse.BooleanOptionalAction, 
                        help="Existing 7z parts won't be downloaded a second time." \
                        "Pass this option whenever download or extraction failed on a first call of the script.")
    args = parser.parse_args()
    dept_code = normalize_dept_code_number(args.dep)
    DownloadRaster(dept_code, args.year, args.resolution, args.retry_download)
