"""

`sgis` is a tool that makes it easier identifying PV panels on building roofs. It is based on Qgis and TensorFlow/Keras. 

It consists in 3 modules:

- `vector_tools`: perform very basic operations on vector layers that describes building cadastre data
- `splitter`: create independant images by intersecting aerial imagery (raster layers) and cadastre data (vector layers)
- `classifier`: define a convolutional model, have it learn some classification skills and apply this classifier on unlabelled images datasets
"""
