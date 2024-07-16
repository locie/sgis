
from qgis.core import edit

from ._utils import context, feedback
from .._init_qgis import processing
from .._utils import get_logger


def add_buffer_distance(layer, distance=4):
    """Increase the area of every feature of a layer by adding a buffer distance.

    Parameters
    ----------
    layer : QgsVectorLayer
        Layer to process.
    distance : float, optional
        Distance, in meters.

    Returns
    -------
    QgsVectorLayer
        Copy of `layer` with modified feature geometries
    """
    logger = get_logger()
    logger.info(f'Vectors buffering, with distance {distance} meters')
    algresult = processing.run(
        "native:buffer",
        {
            'DISSOLVE': False,
            'DISTANCE': distance,
            'END_CAP_STYLE': 2,
            'INPUT': layer,
            'OUTPUT': 'memory:',  # regarding 'memory:' use, see https://gis.stackexchange.com/questions/284064/using-memory-layer-for-processing-algorithms-in-qgis-3
            'JOIN_STYLE': 1,
            'MITER_LIMIT': 2,
            'SEGMENTS': 5
        },
        context=context,
        feedback=feedback,
    )
    return algresult['OUTPUT']



def remove_small_features(layer, min_area=10):
    """Delete every feature whose area is smaller than a given threshold.

    Parameters
    ----------
    layer : QgsVectorLayer
        Layer to process.
    min_area : float, optional
        Feature whose area is smaller than this will be deleted.

    Returns
    -------
    QgsVectorLayer
        Copy of `layer` with some deleted features.
    """
    # computing area: `with_area` is a shallow copy: different attributes but same geometry data/features
    logger = get_logger()
    with_area = processing.run("native:fieldcalculator",
                                {
                                    'INPUT': layer,
                                    'OUTPUT': 'memory:',
                                    'FIELD_NAME': 'area',
                                    'FIELD_TYPE': 0,
                                    'FIELD_LENGTH': 11,  # max 10 million buildings + 3 letters for zip code
                                    'FORMULA': 'area(@geometry)',
                                },
                                context=context,
                                feedback=feedback,
                                )['OUTPUT']

    # selecting features with small areas
    processing.run("qgis:selectbyattribute",
                      {'INPUT': with_area,
                       'FIELD': 'area',
                       'OPERATOR': 5,   # lower or equal
                       'VALUE': f'{min_area}',
                       'METHOD': 0},
                      feedback=feedback,
                      context=context)
    initial_buildings_number = with_area.featureCount()
    unwanted_buildings_number = with_area.selectedFeatureCount()


    # deleting selected features
    with edit(with_area):
        with_area.deleteSelectedFeatures()
    logger.info(
            '{0} buildings out of {1}, smaller than {2} m2, removed'.format(
            unwanted_buildings_number,
            initial_buildings_number,
            min_area
        ))
    return with_area

def add_ID(layer, prefix='', field_length=11):
    """Add a numerical ID as a new field to a layer.

    This ID can be prefixed with a given string.

    Parameters
    ----------
    layer : QgsVectorLayer
        Layer to process.
    prefix : str, optional
        Common part of all features ID, at the beginning of the ID.
    field_length : int, optional
        Number of digits for the ID field, by default 11

    Returns
    -------
    QgsVectorLayer
        Shallow copy of `layer` with an additionnal 'ID' field.


    Notes
    -----
    ID is computed as: ID = concatenation(prefix, id-1) where 'id' is the Qgis id of the feature.
    Thus:

    * ID start from 0.

    * The new field is of type str.
    
    """
    # 3 operations in 1:
    # - auto incremented field [0 based]
    # - conversion of this field to string
    # - addition of a prefix
    # field_length = 11 : # max 10 million buildings + 3 letters for zip code
    logger = get_logger()
    logger.info('Adding IDs')
    algresult = processing.run("native:fieldcalculator",
                    {
                    'INPUT': layer,
                    'FIELD_NAME': 'ID',
                    'FIELD_TYPE': 2,
                    'FIELD_LENGTH': field_length, 
                    'FORMULA': f'concat(\'{prefix}\', to_string(@id-1))',
                    'OUTPUT': 'memory:',

                    },
                    context=context,
                    feedback=feedback,
                    )
    return algresult['OUTPUT'] # shallow copy of `layer`: 1) different attributes 1) same geometry and features



def add_XY_coordinates(layer):
    """Add to a layer 2 new fields for the (X, Y) coordinates.

    Parameters
    ----------
    layer : QgsVectorLayer
        Layer to process.

    Returns
    -------
    QgsVectorLayer
        Shallow copy of `layer` with two new fields: 'X' and 'Y'
    Notes
    -----
    * 'X' and 'Y' field are of type float.
    * Precision is one decimal
    """
    # x0 and y0 reference for Lambert-93: https://fr.wikipedia.org/wiki/Projection_conique_conforme_de_Lambert#/media/Fichier:Lambert_et_mercator_pour_wikipedia.svg
    logger = get_logger()
    logger.info("Adding 'X' field")
    algresult = processing.run("native:fieldcalculator",
                    {'INPUT': layer,
                    'FIELD_NAME':'X',
                    'FIELD_TYPE':0,
                    'FIELD_LENGTH':10,
                    'FIELD_PRECISION':1,
                    'FORMULA': 'x(@geometry)',
                    'OUTPUT': 'memory:'
                    },
                    context=context,
                    feedback=feedback,
                    )

    logger.info("Adding 'Y' field")
    algresult = processing.run("native:fieldcalculator",
                    {'INPUT': algresult['OUTPUT'],
                    'FIELD_NAME':'Y',
                    'FIELD_TYPE':0,
                    'FIELD_LENGTH':10,
                    'FIELD_PRECISION':1,
                    'FORMULA':'y(@geometry)',
                    'OUTPUT': 'memory:'
                    },
                    context=context,
                    feedback=feedback,
                    )
    return algresult['OUTPUT'] # shallow copy: different attributes but same geometry data/features as layer
