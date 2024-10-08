from pandas import read_csv, DataFrame
from PyQt5.QtCore import QVariant
from qgis.core import (QgsGeometry,
                       QgsCoordinateReferenceSystem,
                       edit,
                       QgsField
                       )

from .._init_qgis import processing
from .._utils import get_logger, prepare_paths


def add_protected_buildings(layer, layer_protected_buildings):
    """Quite specific to the French case. Add a boolean attribute, for every building of a vector layer, that states whether 
    this building is included in an architectural protected area.

    Parameters
    ----------
    layer : QgsVectorLayer
        Layer to process.
    layer_protected_buildings : QgsVectorLayer
        Vector layer whose features correspond to protected areas. This layer must be downloaded at http://atlas.patrimoines.culture.fr/atlas/trunk/.

    Returns
    -------
    QgsVectorLayer
        Original `layer` with a new 'protected' boolean attribute that describes whether a building lies in a protected area.

    Notes
    -----
    1. This method must be called before `merge_overlapped_buildings`.

    2. The algorithm proceeds this way:

        1. Both vector files are loaded.

        2. Spatial intersection is performed. 

        3. Buildings for which this intersection is not empty get 'protected=True', other buildings get 'protected=False'.

        4. Useless fields (generated by the intersection) are removed.


    """
    logger = get_logger()

    logger.info("Reprojecting protected buildings")
    result = processing.run("native:reprojectlayer",
                                        {'INPUT': layer_protected_buildings,
                                        'TARGET_CRS': QgsCoordinateReferenceSystem('EPSG:2154'),
                                        'OPERATION':'+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=lcc +lat_0=46.5 +lon_0=3 +lat_1=49 +lat_2=44 +x_0=700000 +y_0=6600000 +ellps=GRS80',
                                        'OUTPUT': 'memory:'})
    layer_protected_buildings_reprojected = result["OUTPUT"]        # no garbage collection, see https://gis.stackexchange.com/questions/284064/using-memory-layer-for-processing-algorithms-in-qgis-3

    logger.info("Joining by location")
    result = processing.run("native:joinattributesbylocation",   # doc: https://docs.qgis.org/3.34/en/docs/user_manual/processing_algs/qgis/vectorgeneral.html#join-attributes-by-location
                                    {'INPUT': layer,
                                    'PREDICATE':[0],
                                    'JOIN': layer_protected_buildings_reprojected,
                                    'JOIN_FIELDS':[],
                                    'METHOD':1,
                                    'DISCARD_NONMATCHING':False,
                                    'PREFIX':'joint_',
                                    'OUTPUT': 'memory:'})
    joint_predictions = result["OUTPUT"]

    logger.info("Adding attribute")
    joint_predictions.dataProvider().addAttributes([QgsField('protected', QVariant.Bool)])
    joint_predictions.updateFields()


    logger.info("Modifying attribute")
    attrs_map = {}
    idx = joint_predictions.fields().indexFromName('protected')

    for feature in joint_predictions.getFeatures():
        id_ = feature['joint_idTigre']
        if id_ is None:
            protected = False
        else:
            protected = True
        attrs_map[feature.id()] = {idx: protected}

    # with edit(joint_predictions):
    #     joint_predictions.dataProvider().changeAttributeValues(attrs_map)
    joint_predictions.dataProvider().changeAttributeValues(attrs_map)


    logger.info("Deleting useless attributes")
    # with edit(joint_predictions):
    #     attr_names = feature.fields().names()
    #     joint_predictions.dataProvider().deleteAttributes([idx for idx, name in enumerate(attr_names) if name.startswith("joint_")])

    attr_names = feature.fields().names()
    joint_predictions.dataProvider().deleteAttributes([idx for idx, name in enumerate(attr_names) if name.startswith("joint_")])
    joint_predictions.updateFields()

    return joint_predictions


def add_roof_type(layer, roof_data_path):
    """Quite specific to the French case. Add to a building vector layer a 'toiture' attribute that describes the most-likely
    roof type (material, color) of the building.

    
    The assigned roof type of building 'k' is the dominant roof type among the buildings of the village of building 'k'.

    Parameters
    ----------
    layer : QgsVectorLayer
        The layer to process. Every feature is a building.
        Must have the following attributes: 
        
        - 'ID': str, identifier of the building

        - 'commune': str, post-code of the village of the building

    roof_data_path : path-like
        CSV file containing the roof type information. Every feature is a building. 
        Must have the following attributes (columns): 
        
        - 'batiment_groupe_id': str
        
        - 'mat_toit_txt': str
        
        The post-code is read as the first 5 characters of 'batiment_groupe_id', and compared to the 'commune' attribute of `layer`.

    Returns
    -------
    pandas.Series
        The share of each roof type in `roof_data_path`, independantly from the commune. For informational purpose only.

    Notes
    -----
    1. The roof type dataset (`roof_data_path`) is an information produced by the Base de Données Nationale des Bâtiments (BDNB). It can be downloaded
       at https://www.data.gouv.fr/en/datasets/base-de-donnees-nationale-des-batiments/. 
       The downloaded files include many other datasets. The roof type information is stored as CSV files ('batiment_groupe_ffo_bat_[...].csv') and can be processed department per department: 
      

    2. Regarding the roof type dataset:

        - Not all buildings and communes are described
        - Whenever an entire commune or the 'mat_toit_txt' field for a building is missing, the 'UNKNOWN' string is used as the roof-type.
    
    3. The reason not to assign the roof type to each building of `layer` is a lack of a common building identifier for `layer` (cadastre data) and roof types (BDNB data).  

    4. The layer is modified in-place, i.e. no copy is returned.

    """
    logger = get_logger()
    roof_data_path = prepare_paths(roof_data_path)
    logger.info('Loading roof type information')
    df = read_csv(roof_data_path,
                     sep=',',
                     engine='c',
                     header=0,
                     usecols=['batiment_groupe_id', 'mat_toit_txt'])

    logger.info('Processing roof type information')

    # post-code: first 5 characters, a priori
    df['batiment_groupe_id'] = df['batiment_groupe_id'].str[:5]
    df.columns = ['commune', 'mat_toit_txt']

    # df = df[df['code_commune'].str.match('\d\d\d\d\d.*')]             # would remove non-numeric post code (e.g. Corsica (2A, 2B), 'uf[...]')

    # filling N.A. values of roof types
    df['mat_toit_txt'] = df['mat_toit_txt'].fillna('UNKNOWN')

    # the roof type of all buildings in a village is the dominant roof type
    sr = df.groupby('commune')['mat_toit_txt'].apply(lambda s: s.value_counts().idxmax())


    vc = sr.value_counts()
    vc = vc / vc.sum()

    try:
        logger.warning(f"{vc['UNKNOWN']:.1%} of commune have an unknown roof type for the majority of their buildings")
    except KeyError as e:
        logger.info(f"0 communes have an unknown roof type for the majority of their buildings")


    logger.info('Merging on commune')
    features = list(layer.getFeatures())
    data = []
    for f in features:
        ID = f['ID']
        commune = f['commune']
        try:
            toiture = sr[commune]
        except KeyError as e:
            logger.warning(f"Missing commune {commune} in the input data, replacing with 'UNKNOWN'")
            toiture = 'UNKNOWN'
        data.append([ID, toiture])
    df_ref = DataFrame(data, columns=['ID', 'toiture']).set_index('ID')

    update_on_ID(layer, df_ref)

    return df.groupby('mat_toit_txt').count().iloc[:,0].sort_values(ascending=False)


def merge_overlapped_buildings(layer, reference_field_name='Score', min_field_value=None):
    """Merge every couple of features whose geometries intersect. 


    Parameters
    ----------
    layer : QgsVectorLayer
        Layer to process. The geometry type must be 'Polygon'.
    reference_field_name : str, optional
        Numerical attribute of `layer` used as a criteria for the merge operation. 
        The attributes of the resulting feature are the one of the feature having the higher value for `reference_field_name`.
        By default 'Score'
    min_field_value : float, optional
        Minimum value that each feature of a merging group must observed for the field `reference_field_name`.

    Notes
    -----
    The layer is modified in-place, i.e. no copy is returned.
    If feature A intersects feature B, and B intersects C then the resulting merged feature includes all of A, B and C geometries. Same for n > 3 intersections.
    If `min_field_value` is specified, intersection between any features A and B is considered 
    if and only if both A and B has a value greater or equal than `min_field_value` for their `reference_field_name` field.
    """
    logger = get_logger()



    logger.info(f"Computing intersections for pairs of buildings")
    if min_field_value is not None:
        logger.info(f"Only features with {reference_field_name} >= {min_field_value} will be considered for merging.")
        layer.setSubsetString(f'"{reference_field_name}" >= {min_field_value}')

        
    result = processing.run("native:multiintersection",
                            {'INPUT': layer,
                                'OVERLAYS': [layer],
                                'OVERLAY_FIELDS_PREFIX': '',
                                'OUTPUT': 'memory:'})
    
    if min_field_value is not None:
        layer.setSubsetString('')

    intersect = result["OUTPUT"]
    intersect.setSubsetString('"ID" != "ID_2"')

    logger.info(f"Creating groups of overlapping buildings")

    # create groups of overlapping buildings
    features = list(intersect.getFeatures())
    features_inter = set()
    for feature in features:
        ID = feature["ID"]
        ID_2 = feature["ID_2"]
        new_couple = (ID, ID_2) if ID < ID_2 else (ID_2, ID)
        features_inter.add(new_couple)

    def is_child(dic, e1, e2):
        '''
        True if: e2 is a child of e1
        False if: e1 not in dict, or e2 not child of e1
        '''
        e = e1
        while e in dic:
            e = dic[e]
            if e == e2:
                return True
        return False

    mapper = {}
    mapper_reverse = {}
    for _, (ID, ID_2) in enumerate(features_inter):
        if not is_child(mapper, ID, ID_2) and not is_child(mapper, ID_2, ID):
            ID_in_mapper = ID in mapper
            if ID_in_mapper:
                k = ID
                while mapper[k] != 'end':  # looking for for the last element of the sequence/group
                    k = mapper[k]
                key = k
            else:
                key = ID

            ID_2_in_mapper = ID_2 in mapper
            if ID_2_in_mapper:
                k = ID_2
                while mapper_reverse[k] != 'start':  #  looking for for the first element of the sequence/group
                    k = mapper_reverse[k]
                value = k
            else:
                value = ID_2

            mapper[key] = value
            mapper_reverse[value] = key

            if not ID_in_mapper:
                mapper_reverse[ID] = 'start'
            if not ID_2_in_mapper:
                mapper[ID_2] = 'end'

    # retrieving groups from dictionary
    group_starts = [k for k, v in mapper_reverse.items() if v == 'start']
    to_edit_features = []
    for group_start in group_starts:
        IDs = []
        k = group_start
        IDs.append(k)
        while mapper[k] != 'end':
            IDs.append(mapper[k])
            k = mapper[k]
        to_edit_features.append(IDs)


    logger.info(f"  {len(features)} intersections of 2 buildings"
                f" leading to {len(to_edit_features)} groups of overlapping buildings")

    # prepare union of geometry and deletion of features, for overlapping buildings
    cpt = 0
    cpt_no_score = 0
    to_delete_features = []
    to_update_geom = {}
    features = layer.getFeatures()
    features_dict = {f["ID"]: f for f in features}
    for IDs in to_edit_features:
        cpt += 1
        score = -1
        leading_feature = None
        all_features = []
        for ID in IDs:
            feature = features_dict[ID]

            all_features.append(feature)
            score_ = feature[reference_field_name]
            if score_ is None:
                logger.warning(f"Feature with ID {ID} has no value for field '{reference_field_name}'. Assuming 0.")
                score_ = 0
                cpt_no_score += 1
            if score_ > score:
                score = score_
                leading_feature = feature
        other_features = [f for f in all_features if f is not leading_feature]
        geom = QgsGeometry.fromWkt('MULTIPOLYGON()')
        for feature in all_features:
            ret_status = geom.addPartGeometry(feature.geometry())
        assert leading_feature is not None
        to_update_geom[leading_feature] = geom
        to_delete_features += other_features

    if cpt_no_score > 100:
        logger.warning(f"More than 100 features had no value for field '{reference_field_name}'.")


    # modify layer
    logger.info(f"Merging")
    layer.setSubsetString("")
    with edit(layer):
        layer.deleteFeatures([feature.id() for feature in to_delete_features])
        for leading_feature, geom in to_update_geom.items():
            _ = layer.changeGeometry(leading_feature.id(), geom)


def update_on_ID(layer, dataframe):
    """Add the data of a DataFrame to the attribute table of a vector layer.

    Columns of the DataFrame become attributes of the layer.

    Parameters
    ----------
    layer : QgsVectorLayer
        Layer to process. Must have an 'ID' attribute that corresponds to the index of `dataframe`.
    dataframe : pandas.DataFrame
        Index of DataFrame must be the ID of the building, with a str dtype.
    Raises
    ------
    NotImplementedError
        If dtype of one column in `dataframe` is not float, int or string.
    Notes
    -----
    Regarding missing elements:
        
        * If some features are in the layer but not in the DataFrame: they are not updated.
        * If some features are in the DataFrame but not in the layer: they are ignored.
    
    `dataframe` is not modified by these operations.
    """
    # todo: possible upgrade: generalize to a left merge (i.e. `layer` as reference) using any other possible column than `ID`
    # --> useful for roof types
    assert 'ID' in layer.fields().names()

    logger = get_logger()

    dataframe = dataframe.copy()
    try:
        dataframe.pop('ID')
    except KeyError as e:
        pass
    else:
        logger.warning("Column 'ID' of `dataframe` is not used for the update.")

    # set real features id's as the DataFrame index
    logger.info('Modifying DataFrame index')
    mapper_id = {}
    for feature in layer.getFeatures():
        mapper_id[feature['ID']] = feature.id()
    dataframe.index = dataframe.index.map(mapper_id)
    dataframe = dataframe[~dataframe.index.isna()]          # ignore features that do not correspond to a feature ID

    # add a new attribute for each column of the DataFrame
    logger.info('Adding new attributes to layer')
    attributes = []
    for col, dtype in dataframe.dtypes.items():
        if dtype == 'O':
            Qt_type = QVariant.String
        elif ('int' in str(dtype)):
            Qt_type = QVariant.Int
        elif ('float' in str(dtype)):
            Qt_type = QVariant.Double
        else:
            raise NotImplementedError(f'{dtype} dtype of column {col} is not supported')
        attributes += [QgsField(col, Qt_type)]
    layer.dataProvider().addAttributes(attributes)
    layer.updateFields()

    # replace the name of the DataFrame columns by the index of the corresponding attribute in `layer`
    logger.info('Modifying DataFrame columns')
    dataframe.columns = [layer.fields().indexFromName(col_name) for col_name in dataframe.columns]

    # update the features
    logger.info('Updating features')
    attrs_update = dataframe.to_dict(orient='index')
    layer.dataProvider().changeAttributeValues(attrs_update)


    ## below: features update without using dataProvider(). Instead, high level context manager `edit(layer)` is used
    # with edit(layer):
    #     for feature in layer.getFeatures():
    #         id_ = feature.id()
    #         try:
    #             data = dataframe.loc[id_]
    #         except KeyError as e:
    #             pass
    #         else:
    #             for col_name, value in data.to_dict().items():  # `to_dict` returns Python types, contrary to `items()` that returns Numpy dtypes
    #                 feature[col_name] = value
    #                 layer.updateFeature(feature)
