"""
Join the prediction scores information (CSV, result of CNN model) with the cadastral geometry information (SHP, Etalab).

example:

activate_PV_detection;
export PYTHONPATH=~/sgis:~/sgis/src:~/sgis/production_scripts:$PYTHONPATH
dep=75;year=2011;model=2C_22_34_35_67_73__V14;epoch=5;roof_type=false;merge_overlapping=pv;export_shp=true;
python ~/sgis/production_scripts/postprocess.py --dep $dep --year $year --model $model --epoch $epoch --roof_type ${roof_type} --merge_overlapping ${merge_overlapping} --export_shp ${export_shp}
"""

#!/usr/bin/env python
# coding: utf-8

# **chaque booléen définit un suffixe équivalent dans le nom des fichiers exportés**

import argparse
from sgis import *
import pandas as pd
from pathlib import Path
from sgis.vector_tools import *
from os import environ
from production_scripts._production_constants import DEFAULT_CLASSIFIER_MODEL

def main(dep, year, name, roof_type, protected_buildings, merge_overlapping, export_shp_, epoch, model = DEFAULT_CLASSIFIER_MODEL, base_folder_path = None):
    epoch = f"{epoch:0>3d}"
    assert ((len(dep)==2) or (len(dep)==3 and dep[0]=='9'))

    #  construit les chemins d'entree/sortie
    if(base_folder_path is None):
        base_folder_path = Path(environ['HOME'])
    else:
        base_folder_path = Path(base_folder_path)
        
    path_predictions =      base_folder_path / "predictions" / dep / year / rf"{model}_{epoch}/raw_predictions/scores.csv"
    path_output =           base_folder_path / "predictions" / dep / year / rf"{model}_{epoch}/prediction_postprocessing/"
    # path_predictions =      base_folder_path + f'/scores_34.csv'
    # path_output =           base_folder_path + f'/postprocessing_34/'
    path_preprocessing =    base_folder_path  / "split"/ dep / year / rf"preprocessing/vectors/batiments.shp"

    path_output = Path(path_output)
    path_output.mkdir(exist_ok=True, parents=True)
    suffix = ''

    ##################### Lecture cadastre
    # Lecture score
    scores = pd.read_csv(path_predictions, header=0, index_col=0, names=['Score', 'Class']).drop('Class', axis=1)
    scores.index = scores.index.astype(str)  # 
    index_ex = scores.index[0]
    print(f"Typical index is: {index_ex}")
    

    if any([e[0] in range(0, 10) for e in scores.index]):
        raise NotImplementedError("Cas 'ID' détecté. " \
                                  "Pas de disjonction de cas ID / rnb_id, " \
                                  "nécessaire pour ajouter un '0' en préfixe dans le cas ID.")
    
 
    # try:
    #     float(index_ex)          # note: possible car les résultats de prédiction pour les bâtiments  type XXX123_1.jpg ne sont pas conservés
    # except:                             # cas rnb_id ou cas particulier etalab où l'index est interprété comme str (ex Corse (2A, 2B)): 
    #     p = ''                             # la lecture du CSV a conservé les 0 ('02A', '02B')    
    # else:                               # l'index a été interprété comme un entier, 
    #                                     # donc les 0 sont perdus     
    #     if len(dep) == 2:
    #         if dep[0] == '0':   # ex: dep=2
    #             p = '00'     
    #         else:               # ex: dep=45
    #             p = '0'
    #     else:                   # ex: dep = 972
    #         p = ''
                
    # scores.index = p + scores.index 
    # index_ex = scores.index[0]
    # print(f"    Added prefix for prediction score: '{p}'")
    # print(f"    Typical index will now be: {index_ex}")
    # print(f"    Proceeding in 10 seconds") 

    # Ajout scores au cadastre
    with VectorTools() as qgis_inst:
        layer = qgis_inst.load_layer(path_preprocessing, f'postprocessed_{dep}')
        layer = qgis_inst.copy_layer(layer)
        qgis_inst.update_on_ID(layer, scores)
        suffix += "score"

        # vérification que les scores sont bien définis (succès de l'update précédent)
        one_central_feature_index = layer.featureCount()//2
        attributes = layer.getFeature(one_central_feature_index).attributeMap()
        print("Sample of attributes of preprocessing layer with prediction scores:")
        print(attributes)
        if not (attributes['Score']>=0):
            raise ValueError("Merging process went probably wrong since there are some missing prediction values. Check the compatibility of the preprocessed layer and prediction CSV.")

        # Toiture
        if roof_type:
            raise NotImplementedError("")
            dep_ = dep.lstrip('0') # remove leading 0 if dep is 1-digit (1->9)
            path_roof_types = base_folder_path / rf"LaCie_thebaulm/gis/other/roof_types/batiment_groupe_ffo_bat_{dep_}.csv"
            roof_per_building = qgis_inst.add_roof_type(layer, path_roof_types)
            
            roof_per_building.name = 'Number of buildings'
            roof_per_building.to_csv(path_output / 'roof_types_buildings_repartition.csv')
            
            suffix += "_roof"

        # Batiments protégés
        if protected_buildings:    
            path_protected_buildings = base_folder_path / f"LaCie_thebaulm/gis/vectors/protected_buildings/ProtectionautitredesabordsdemonumentshistoriquesAC1{name}{dep}/Documents/supmh{dep}_exporttigrepolygone.shp"
            layer_protected_buildings = qgis_inst.load_layer(path_protected_buildings, 'protected_buildings')
            layer = qgis_inst.add_protected_buildings(layer, layer_protected_buildings)
            suffix += "_protected"

        # Bâtiments superposés
        if merge_overlapping:
            if merge_overlapping.lower() == 'pv':
                min_field_value = 0.5
            else:
                assert merge_overlapping.lower() == 'all'
                min_field_value = None
            suffix += f"_merge{merge_overlapping.lower().capitalize()}"
            qgis_inst.merge_overlapped_buildings(layer, min_field_value=min_field_value)

        if export_shp_:
            qgis_inst.export_shp(layer, path_output, suffix)    
        qgis_inst.export_csv(layer, path_output / f'{suffix}.csv')

    # verification
    from subprocess import run
    print("\n\nFile excerpt for verification:")
    run(f"tail {path_output / f'{suffix}.csv'}", shell=True)
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dep", type=str, required=True, help="Department code: 2 digits from 01 to 99 else 3 digits")
    parser.add_argument("--year", type=str, required=True, help='BDOrtho version [YYYY]')
    parser.add_argument("--model", required=True, default=DEFAULT_CLASSIFIER_MODEL, help="Name of the CNN model, without extension, without epoch")
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
    
    main(dep, year, name, roof_type, protected_buildings, merge_overlapping, export_shp_,  epoch, model)

    
