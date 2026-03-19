from dataclasses import dataclass
from pathlib import Path
from production_scripts.rnb_geo_api import find_dept_metadata, request_all_rnb_csv_metadata

BASE_ETALAB_URL="https://cadastre.data.gouv.fr/data/etalab-cadastre" # ETALAB
BASE_RNB_URL="https://rnb-opendata.s3.fr-par.scw.cloud/files" # R.N.B. : Référentiel National des Bâtiments


@dataclass
class CadastreData:
    dept_code: str
    date: str
    raw_dirname: str
    url: str
    zip_filename: str
    output_filename: str
    expected_hash: str

    def build_paths(self, base_dir: Path):
        rootp = base_dir / self.date
        zipped_dir = rootp / "zipped"
        unzipped_dir = rootp / "unzipped" / self.raw_dirname

        zipped_dir.mkdir(parents=True, exist_ok=True)
        unzipped_dir.mkdir(parents=True, exist_ok=True)

        zip_file_path = zipped_dir / self.zip_filename
        unzipped_file_path = unzipped_dir / self.output_filename

        return zip_file_path, unzipped_file_path
    
@dataclass
class Etalab(CadastreData):

    @classmethod    
    def from_dep(cls, dept_code: str, date: str) -> "Etalab":
        raw_dirname = f"cadastre-{dept_code}-batiments-shp"
        return cls(
            dept_code=dept_code,
            date=date,
            raw_dirname=raw_dirname,
            url=rf"{BASE_ETALAB_URL}/{date}/shp/departements/{dept_code}/cadastre-{dept_code}-batiments-shp.zip",
            zip_filename=f"{raw_dirname}.zip",
            output_filename="batiments.shp",
            expected_hash=None #TODO Actuellement pas de sha1 disponible sur le site web de ETALAB
        )
    
   
    
@dataclass
class RNB(CadastreData):

    @classmethod
    def from_dep(cls, dept_code: str) -> "RNB":
        raw_dirname = f"cadastre-{dept_code}-batiments-csv"
        f_stem = rf"RNB_{dept_code}.csv"
        zip_filename=f"{f_stem}.zip"
        url = rf"{BASE_RNB_URL}/{zip_filename}"

        all_dept_metadata = request_all_rnb_csv_metadata()
        dep_metadata = find_dept_metadata(all_dept_metadata, int(dept_code))        
        return cls(
            dept_code=dept_code,
            date=dep_metadata.date,
            raw_dirname=raw_dirname,
            url=url,
            output_filename=f_stem,
            zip_filename=zip_filename,
            expected_hash=dep_metadata.sha1,
        )
    
