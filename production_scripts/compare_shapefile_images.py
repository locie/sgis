import re
from os import environ
from pathlib import Path
from qgis.core import QgsVectorLayer, QgsFeatureRequest, QgsApplication


display = environ.get("DISPLAY")
if not display:# No graphical display available
    environ["QT_QPA_PLATFORM"] = "offscreen"
    # cas d'absence de session X (i.e. pas de support Qt)
    # solution: déclarer une variable d'env: os.environ["QT_QPA_PLATFORM"] = "offscreen"
    
QgsApplication.setPrefixPath("/usr", True)   # à adapter à ton installation
qgs = QgsApplication([], False)
qgs.initQgis()

shp = Path(r"/home/thebaulm/split/15/2025/preprocessing/vectors/batiments.shp")
img_dir = Path(r"/home/thebaulm/split/15/2025/rasters/images")

layer = QgsVectorLayer(str(shp), "batiments", "ogr")
if not layer.isValid():
    raise RuntimeError(f"Impossible d'ouvrir {shp}")

id_col = next((c for c in ("ID", "rnb_id") if c in layer.fields().names()), None)
if id_col is None:
    raise KeyError("Pas de champ 'ID' ni 'rnb_id'")

req = (QgsFeatureRequest()
       .setFlags(QgsFeatureRequest.NoGeometry)
       .setSubsetOfAttributes([id_col], layer.fields()))
ids = {str(f[id_col]) for f in layer.getFeatures(req)}

surnumeraires = []   # ID absent du shapefile (même après retrait du suffixe)
suffixees = []       # ID présent, mais image avec suffixe _N

for p in sorted(img_dir.glob("*.jpg")):
    stem = p.stem
    if stem in ids:
        continue
    base = re.sub(r"_\d+$", "", stem)
    if base != stem and base in ids:
        suffixees.append(p.name)
    else:
        surnumeraires.append(p.name)

print(f"Images surnuméraires (ID absent du shapefile) : {len(surnumeraires)}")
for n in surnumeraires:
    print("  ", n)

print(f"\nImages suffixées (doublons de rasters, ID valide) : {len(suffixees)}")

Path("images_surnumeraires.txt").write_text("\n".join(suffixees))
qgs.exitQgis()