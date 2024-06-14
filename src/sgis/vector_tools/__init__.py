"""A small intro to `vector_tools`
"""
from ._utils import copy_layer, export_csv, export_shp, load_layer
from ._preprocessing import add_buffer_distance, remove_small_features, add_ID, add_XY_coordinates
from ._external_data import add_protected_buildings, add_roof_type, merge_overlapped_buildings, update_on_ID
from .._utils import clean_processing_folder

__all__ = ['copy_layer', 'export_csv', 'export_shp', 'load_layer', 
           'add_buffer_distance', 'remove_small_features', 'add_ID', 'add_XY_coordinates',
           'add_protected_buildings', 'add_roof_type', 'merge_overlapped_buildings', 'update_on_ID',
           'clean_processing_folder']  # todo: need for doc automodule directive
