"""

`sgis` is a tool that makes it easier identifying PV panels on building roofs. It is based on Qgis and TensorFlow/Keras. 

It consists in 3 modules:

- `vector_tools`: perform very basic operations on vector layers that describes building cadastre data
- `splitter`: create independant images by intersecting aerial imagery (raster layers) and cadastre data (vector layers)
- `classifier`: define a convolutional model, have it learn some classification skills and apply this classifier on unlabelled images datasets
"""
import warnings
__all__ = ['vector_tools','splitter','classifier']

from . import vector_tools

# Filtre warning tensorflow: DeprecationWarning: `np.bool8` is a deprecated alias for `np.bool_`.  (Deprecated NumPy 1.24)
warnings.filterwarnings(
    "ignore",
    message=r".*np\.bool8.*deprecated.*",
    category=DeprecationWarning
)
# Filtre warning tensorflow : DeprecationWarning: the imp module is deprecated in favour of importlib and slated for removal in Python 3.12; see the module's documentation for alternative uses mod = _builtin_import(name, globals, locals, fromlist, level)
warnings.filterwarnings(
    "ignore",
    message=r".*the imp module.*deprecated.*",
    category=DeprecationWarning
)