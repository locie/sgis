from qgis.core import QgsApplication , QgsProcessingContext, QgsProcessingFeedback
import processing from processing.core.Processing import Processing

qgs = QgsApplication([], False)
qgs.initQgis()

Processing.initialize()

feedback = QgsProcessingFeedback()
context = QgsProcessingContext()

# for alg in QgsApplication.processingRegistry().algorithms():
#     print(alg.id(), "->", alg.displayName())

# from qgis.core import (
#     QgsApplication,
#     QgsProcessingContext,
#     QgsProcessingFeedback
# )

# from processing.core.Processing import Processing


# class QGISApp:
#     def __init__(self, prefix="/usr"):
#         # MUST be first
#         QgsApplication.setPrefixPath(prefix, True)

#         self.qgs = QgsApplication([], False)
#         self.qgs.initQgis()

#         # Processing AFTER initQgis
#         Processing.initialize()

#         self.context = QgsProcessingContext()
#         self.feedback = QgsProcessingFeedback()

#     def list_algorithms(self):
#         for alg in QgsApplication.processingRegistry().algorithms():
#             print(alg.id(), "->", alg.displayName())

#     def close(self):
#         QgsApplication.exitQgis()


# if __name__ == "__main__":
#     app = QGISApp()
#     app.list_algorithms()
#     app.close()