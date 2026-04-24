from os import environ
import shutil
from qgis.core import (
            QgsApplication,
            QgsProcessingContext,
            QgsProcessingFeedback,
            QgsProject
        )
from processing.core.Processing import Processing #bootstrap manager for QGIS Processing.

display = environ.get("DISPLAY")
if not display:# No graphical display available
    environ["QT_QPA_PLATFORM"] = "offscreen"
    # cas d'absence de session X (i.e. pas de support Qt)
    # solution: déclarer une variable d'env: os.environ["QT_QPA_PLATFORM"] = "offscreen"

class QgisManager():
   
    def __enter__(self, prefix=None):          
        # MUST be first
        # Without this, QGIS guesses paths. In debug runs the environment is often cleaner, so it “works”.
        # In normal runs → provider registry loads garbage → 💥 segfault.
        if(prefix == None):
            prefix = shutil.which("qgis")
            print(prefix)
        
        QgsApplication.setPrefixPath(prefix, True)

        self.qgs = QgsApplication([], False)
        self.qgs.initQgis()
        
        Processing.initialize()     
        self.context = QgsProcessingContext()
        self.feedback = QgsProcessingFeedback()
        self.qjsproject = QgsProject.instance()
            
        return self
    
    def catch_qgis_messages_enable(self):
        from qgis.core import Qgis
        
        # referencing before assignment
        def _catch_qgis_messages(message, tag, level):
            if level == Qgis.Info: type="Info"
            elif level == Qgis.Warning: type="Warning"
            elif level == Qgis.Critical: type="Critical"
            elif level == Qgis.Success: type="Success"
            else: type="unknown message type"
            print(f"[QGIS] - [{type}] - {tag}: {message}")
        # enable catching qgis messages
        print("QGIS messageLog is now connected.")                  
        self.qgs.messageLog().messageReceived.connect(_catch_qgis_messages)
      
    def list_algorithms(self):      
        for alg in self.qgs.processingRegistry().algorithms():
            print(alg.id(), "->", alg.displayName())

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.qgs.exit() # use exit() instead of exitQgis()
        """NOTE use app.exit() instead of app.exitQgis().
        QGIS is not designed for multiple init/exit cycles in one interpreter.
        NOT call exitQgis() again in the same process.
        Because:
        - GDAL driver manager is global
        - Qt application object may persist
        - Static C++ singletons inside QGIS do not fully reset
        - SIP bindings don't reinitialize cleanly
        This is a design limitation of QGIS + Qt + GDAL.
        """
             
if __name__ == "__main__":
    app = QgisManager()
    app.list_algorithms()