import sys

# if it is going to be executed on Google Colab, it will be needed to add the location of the virtual environment into
# the path variable

sys.path.append(
    '/content/drive/MyDrive/Colab Notebooks/PoliTo: Thesis/modality-invariance-VLMs/VLPs_env/lib/python3.10/site-packages/'
    )

import os
import csv
import numpy as np
from typing import List

from utils.arg_parser import parse_args
from utils.multimodal_similarities import multimodal_similarities
from utils.save_objects import save_objects
from utils.save_data import save_data
from utils.chart_utils import boxplot, tsne_2dplot
from utils.scatter import scatter

from models.custom_clip import CustomCLIP
from models.custom_align import CustomALIGN
from models.custom_imagebind import CustomImageBind
from models.custom_cyclip import CustomCyCLIP
from models.custom_flava import CustomFLAVA
from models.custom_albef import CustomALBEF


def create_path_if_not_existant(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':

    model_name2model = {
    # 'CLIP_ViT-B32' : CustomCLIP('CLIP_ViT-B32'),
    # 'CLIP_RN50'    : CustomCLIP('CLIP_RN50'),
    # 'ALIGN'        : CustomALIGN(),
    # 'ImageBind'    : CustomImageBind(),
    # 'CyCLIP'       : CustomCyCLIP(),
    # 'FLAVA'        : CustomFLAVA(),
    'ALBEF'        : CustomALBEF()
    }
    
    # Parse the command-line arguments
    args = parse_args()

    # Assign values to variables
    model_name = args.MODELNAME
    test_dataset = args.DATASET
    image_root_path = args.IMAGEROOTPATH

    create_path_if_not_existant('results')
    outfile = 'cmd.csv'

    if not os.path.exists(os.path.join('results',outfile)):
        with open(os.path.join('results',outfile), 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(['tested dataset', 'model name', 'Txt-Txt', 'Img-Img', 'Img-Txt'])

    assert test_dataset in ['mscoco',
                            'flickr30k',
                            'amazon_products',]

    assert model_name in ['CLIP_ViT-B32',
                          'CLIP_RN50',
                          'ALIGN',
                          'ImageBind',
                          'CyCLIP',
                          'FLAVA',
                          'ALBEF',]

    model = model_name2model[model_name]

    all_sim_img, all_dissim_img,\
        all_sim_txt, all_dissim_txt,\
            all_sim_txtimg,\
                all_dissim_txtimg = multimodal_similarities(model,test_dataset,image_root_path,)

    pos_type = ['Txt-Txt']*len(all_sim_txt)
    pos_type.extend(['Img-Img']*len(all_sim_img))
    pos_type.extend(['Txt-Img']*len(all_sim_txtimg))
    all_sim = np.concatenate((all_sim_txt, all_sim_img, all_sim_txtimg))
    pos_gobj = boxplot(all_sim, pos_type, 'Pos', 'dodgerblue')

    neg_type = ['Txt-Txt']*len(all_dissim_txt)
    neg_type.extend(['Img-Img']*len(all_dissim_img))
    neg_type.extend(['Txt-Img']*len(all_dissim_txtimg))
    all_dissim = np.concatenate((all_dissim_txt, all_dissim_img, all_dissim_txtimg))
    neg_gobj = boxplot(all_dissim, neg_type, 'Neg', 'indianred')

    outfolder = os.path.join('results', 'charts', model_name, test_dataset,)
    create_path_if_not_existant(outfolder)

    save_objects([pos_gobj, neg_gobj], outfolder, name='sim_distrib')

    save_data([model_name]*len(np.concatenate((all_sim, all_dissim))),
              [test_dataset]*len(np.concatenate((all_sim, all_dissim))),
              np.concatenate((all_sim, all_dissim)),
              np.concatenate((pos_type, neg_type)),
              np.concatenate((['Pos']*len(all_sim), ['Neg']*len(all_dissim))),
              outfolder,
              'raw_distrib',)
    
    for i in range(5):
        scatter(model, test_dataset, outfolder, image_root_path, i+1,)