from os import environ
display = environ.get("DISPLAY")
if not display:# No graphical display available
    environ["QT_QPA_PLATFORM"] = "offscreen"
    # cas d'absence de session X (i.e. pas de support Qt)
    # solution: déclarer une variable d'env:
    #     os.environ["QT_QPA_PLATFORM"] = "offscreen"
            
from qgis.core import (
    QgsApplication,
    QgsProcessingContext,
    QgsProcessingFeedback
)
from processing.core.Processing import Processing #bootstrap manager for QGIS Processing.
from ._utils import load_layer, copy_layer, export_csv, export_shp
        

class QgisManager():
    def __init__(self, prefix="/usr"):   
        # MUST be first
        # Without this, QGIS guesses paths. In debug runs the environment is often cleaner, so it “works”.
        # In normal runs → provider registry loads garbage → 💥 segfault.
        QgsApplication.setPrefixPath(prefix, True)

        self.qgs = QgsApplication([], False)
        self.qgs.initQgis()

        # Processing AFTER initQgis
        Processing.initialize()

        self.context = QgsProcessingContext()
        self.feedback = QgsProcessingFeedback()

    def list_algorithms(self):
        for alg in QgsApplication.processingRegistry().algorithms():
            print(alg.id(), "->", alg.displayName())

    def close(self):
        # MUST be last
        QgsApplication.exitQgis()
        
if __name__ == "__main__":
    app = QgisManager()
    app.list_algorithms()
    app.close()