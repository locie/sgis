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
    QgsProcessingFeedback,
    Qgis
)
from processing.core.Processing import Processing #bootstrap manager for QGIS Processing.
        

class QgisManager():
    def __init__(self, prefix="/thebaulm"):   
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
    
    def catch_qgis_messages_enable(self):
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

    def _close(self):
        # MUST be last
        self.qgs.exitQgis()
        
    def __del__(self):
        self._close() 
             
if __name__ == "__main__":
    app = QgisManager()
    app.list_algorithms()