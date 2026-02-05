from os import environ
display = environ.get("DISPLAY")
if not display:# No graphical display available
    environ["QT_QPA_PLATFORM"] = "offscreen"
    # cas d'absence de session X (i.e. pas de support Qt)
    # solution: déclarer une variable d'env:
    #     os.environ["QT_QPA_PLATFORM"] = "offscreen"

from ._qgis_manager import QgisManager # base class
from ._utils import QgisUtils # extra class
from ._preprocessing import QgisPreprocessing # extra class
from ._external_data import QgisExternalData # extra class


# Final class VectorTools that inherits (QgisManager, QgisUtils, QgisPreporcessing, QgisExternalData)
class VectorTools(QgisManager, QgisUtils, QgisPreprocessing, QgisExternalData):
    pass