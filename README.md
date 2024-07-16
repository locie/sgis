**sgis** is a tool that makes it easier identifying PV panels on building roofs. It is based on Qgis and TensorFlow/Keras.

# Content

It consists in 3 modules:

    - vector_tools: perform very basic operations on vector layers that describes building cadastre data

    - splitter: create independant images by intersecting aerial imagery (raster layers) and cadastre data (vector layers)

    - classifier: define a convolutional model, have it learn some classification skills and apply this classifier on unlabelled images datasets


# Installation

`sgis` relies on 2 large python packages:

- `qgis` and its python API
- `tensorflow`

The recommended way is to set up every dependencies is to use a dedicated Anaconda environment. If needed, please install `tensorflow` using _pip_ within this environment.
Once the environment is created, please clone this repo and install manually using pip within your environment. For instance:

```
git clone <this_repo>
conda activate my_env
pip install -e /path/to/cloned/repo
```


# Documentation

Please open `doc/build/html/index.html' with a web browser.

# Contact

If anything is going wrong or you are looking for more information, please contact Boris Nerot at boris.nerot@univ-smb.fr.

