"""
Join the prediction scores information (CSV, result of CNN model) with the cadastral geometry information (SHP, Etalab).

example:

activate_PV_detection;export QT_QPA_PLATFORM=offscreen;
dep=75;year=2011;model=2C_22_34_35_67_73__V14;epoch=5;roof_type=true;merge_overlapping=pv;export_shp=true;
python ~/sgis/production_scripts/postprocess.py --dep $dep --year $year --model $model --epoch $epoch --roof_type ${roof_type} --merge_overlapping ${merge_overlapping} --export_shp ${export_shp}
"""

#!/usr/bin/env python
# coding: utf-8

# **chaque booléen définit un suffixe équivalent dans le nom des fichiers exportés**

import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dep", type=str, required=True, help="Department code: 2 digits from 01 to 99 else 3 digits")
parser.add_argument("--year", type=str, required=True, help='BDOrtho version [YYYY]')
parser.add_argument("--model", required=True, default='2C_22_34_35_67_73__V14', help="Name of the CNN model, without extension, without epoch")
parser.add_argument("--epoch", required=True, type=int, help="Training epoch to be used. Refines the model argument.")
parser.add_argument("--name", type=str, help='Department name - required for protected_buildings')
parser.add_argument("--roof_type", type=str, default='true', choices=['true', 'false'], help='Whether the mean roof type of the city is added to each building')
parser.add_argument("--protected_buildings", type=str, default='false', choices=['true', 'false'], help="If 'true', states whether each building is located in a protected heritage area")
parser.add_argument("--merge_overlapping", choices=['pv', 'all'], type=str, help='How to merge overlapped buildings. No merge is done if unprovided.')
parser.add_argument("--export_shp", type=str, choices=['true', 'false'], default='false', help='Whether vector layer must be exported, in addition to csv file')
args = parser.parse_args() 


def litt_eval(arg):
    if arg.lower() == 'true':
        return True
    else:
        assert arg.lower() == 'false'
        return False

dep = args.dep
year = args.year
name = args.name
roof_type = litt_eval(args.roof_type)
protected_buildings = litt_eval(args.protected_buildings)
merge_overlapping = args.merge_overlapping
export_shp_ = litt_eval(args.export_shp)
model = args.model
epoch = args.epoch

epoch = f"{epoch:0>3d}"



assert ((len(dep)==2) or (len(dep)==3 and dep[0]=='9'))

from sgis import *
import pandas as pd
from pathlib import Path
from sgis.vector_tools import *
from time import sleep
from os import environ






home = environ['HOME']
path_predictions =      home + f'/predictions/{dep}/{year}/{model}_{epoch}/raw_predictions/scores.csv'
path_output =           home + f'/predictions/{dep}/{year}/{model}_{epoch}/prediction_postprocessing/'
# path_predictions =      home + f'/scores_34.csv'
# path_output =           home + f'/postprocessing_34/'
path_preprocessing =    home + f'/split/{dep}/{year}/preprocessing/vectors/batiments.shp'

path_output = Path(path_output)
path_output.mkdir(exist_ok=True, parents=True)
suffix = ''




##################### Lecture cadastre

layer = load_layer(path_preprocessing, f'postprocessed_{dep}')
layer = copy_layer(layer)



# Lecture score


scores = pd.read_csv(path_predictions, header=0, index_col=0, names=['Score', 'Class']).drop('Class', axis=1)
scores.index = scores.index.astype(str)  # 
index_ex = scores.index[0]
print(f"Typical index is: {index_ex}")
try:
    float(index_ex)          # note: possible car les résultats de prédiction pour les bâtiments  type XXX123_1.jpg ne sont pas conservés
except:                             # cas particulier où l'index est interprété comme str (ex Corse (2A, 2B)): 
                                    # la lecture du CSV a conservé les 0 ('02A', '02B')
    p = ''
else:                               # l'index a été interprété comme un entier, 
                                    # donc les 0 sont perdus     
    if len(dep) == 2:
        if dep[0] == '0':   # ex: dep=2
            p = '00'     
        else:               # ex: dep=45
            p = '0'
    else:                   # ex: dep = 972
        p = ''

    

scores.index = p + scores.index 
index_ex = scores.index[0]
print(f"    Added prefix for prediction score: '{p}'")
print(f"    Typical index will now be: {index_ex}")

print(f"    Proceeding in 10 seconds") 
sleep(10)



# Ajout scores au cadastre



update_on_ID(layer, scores)

suffix += "score"


# vérification que les scores sont bien définis (succès de l'update précédent)
attributes = layer.getFeature(20000).attributeMap()
print("Sample of attributes of preprocessing layer with prediction scores:")
print(attributes)
if not (attributes['Score']>=0):
    raise ValueError("Merging process went probably wrong since there are some missing prediction values. Check the compatibility of the preprocessed layer and prediction CSV.")



# Toiture




if roof_type:
    dep_ = dep.lstrip('0') # remove leading 0 if dep is 1-digit (1->9)
    path_roof_types = Path(rf'{home}/LaCie_thebaulm/gis/other/roof_types') / f'batiment_groupe_ffo_bat_{dep_}.csv'
    roof_per_building = add_roof_type(layer, path_roof_types)
    
    roof_per_building.name = 'Number of buildings'
    roof_per_building.to_csv(path_output / 'roof_types_buildings_repartition.csv')
    
    suffix += "_roof"


# Batiments protégés



if protected_buildings:    
    path_protected_buildings = Path(rf'{home}/LaCie_thebaulm/gis/vectors/protected_buildings/') \
                               / fr'ProtectionautitredesabordsdemonumentshistoriquesAC1{name}{dep}'/'Documents'/f'supmh{dep}_exporttigrepolygone.shp'
    layer_protected_buildings = load_layer(path_protected_buildings, 'protected_buildings')
    layer = add_protected_buildings(layer, layer_protected_buildings)
    suffix += "_protected"


# Bâtiments superposés



if merge_overlapping:
    if merge_overlapping.lower() == 'pv':
        min_field_value = 0.5
    else:
        assert merge_overlapping.lower() == 'all'
        min_field_value = None
    suffix += f"_merge{merge_overlapping.lower().capitalize()}"
    merge_overlapped_buildings(layer, min_field_value=min_field_value)

if export_shp_:
    export_shp(layer, path_output, suffix)    
export_csv(layer, path_output / f'{suffix}.csv')


# verification
from subprocess import run
print("\n\nFile excerpt for verification:")
run(f"tail {path_output / f'{suffix}.csv'}", shell=True)
print("\n")
