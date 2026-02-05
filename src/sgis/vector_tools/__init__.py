from _qgis_manager import QgisManager # base class
from _utils import QgisUtils # extra class
from _preprocessing import QgisPreporcessing # extra class
from _external_data import QgisExternalData # extra class


# Final class vector_tools that inherits (QgisManager, QgisUtils, QgisPreporcessing)
class VectorTools(QgisManager, QgisUtils, QgisPreporcessing, QgisExternalData):
    pass