from qgis.core import QgsApplication , QgsProcessingContext, QgsProcessingFeedback


qgs = QgsApplication([], False)
qgs.initQgis()

import processing
from processing.core.Processing import Processing
Processing.initialize()

feedback = QgsProcessingFeedback()
context = QgsProcessingContext()

# for alg in QgsApplication.processingRegistry().algorithms():
#     print(alg.id(), "->", alg.displayName())