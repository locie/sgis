from production_scripts import download_cadastre_from_rnb_api as rnbapi

dept_code=75
limit=100
bbox='5.33, 45.00, 7.15, 46.00' 

rnbapi.get_dept_buildings_geojson(dept_code)
rnbapi.get_bbox_buildings_geojson(bbox, limit)