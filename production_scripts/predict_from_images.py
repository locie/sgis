# exemple appel
"""
Apply a prediction model to split images of a given department and year.

ex:
dep=75;year=2011;model=2C_22_34_35_67_73__V14;epoch=5;
python ~/sgis/production_scripts/predict_from_images.py --dep $dep --year $year --model $model --epoch $epoch --share 0.9999999
"""
import argparse
from os import environ
from sgis.classifier import *

DEFAULT_SHARE_VALUE = 0.99999
DEFAULT_MODEL = '2C_22_34_35_67_73__V14'
DEFAULT_GPU = 'true'

def main(dep, year, epoch, model = DEFAULT_MODEL, share = DEFAULT_SHARE_VALUE, gpu = DEFAULT_GPU ):
    epoch = f"{epoch:0>3d}"
    assert gpu in ('true', 'false')
    if gpu == 'false':  
        environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print(f'Prediction on department {dep} (year {year}) with share {share}')
    home = '/tmp/sgis/unittests'
    my_model = load_model(rf'{home}/classification/models/{model}/saved_models/{model}.metadata', 
                        rf'{home}/classification/models/{model}/saved_models/{model}_{epoch}.keras')

    input_path =  home + f'/split/{dep}/{year}/rasters/'
    output_path = home + f'/predictions/{dep}/{year}/{model}_{epoch}/raw_predictions/'

    predictions = my_model.predict(input_path, output_path, share=share, copy_images=False, save_scores=True)
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dep", type=str, required=True, help="Department code: 2 digits from 01 to 99 else 3 digits")
    parser.add_argument("--year", type=str, required=True, help='BDOrtho version [YYYY]')
    parser.add_argument("--model", required=True, default=DEFAULT_MODEL, help="Name of the CNN model, without extension, without epoch")
    parser.add_argument("--epoch", required=True, type=int, help="Training epoch to be used. Refines the model argument.")
    parser.add_argument("--share", default=DEFAULT_SHARE_VALUE, type=float, help="Share of available images to apply prediction on. Must be different than 1.")
    parser.add_argument("--gpu", default=DEFAULT_GPU, choices=('true', 'false'), help="Whether the machine GPU is used for prediction")
    args = parser.parse_args()

    dep = args.dep
    year = args.year
    share = args.share
    gpu = args.gpu
    model = args.model
    epoch = args.epoch
    
    main(dep, year, epoch, model, share, gpu)