import sys

# if it is going to be executed on Google Colab, it will be needed to add the location of the virtual environment into
# the path variable

sys.path.append(
    '/content/drive/MyDrive/Colab Notebooks/PoliTo: Thesis/modality-invariance-VLMs/VLPs_env/lib/python3.10/site-packages/'
    )

import os
# import csv
# import numpy as np

import torch

from utils.arg_parser import parse_args
from utils.datasets.flickr30k_captions import Flickr30kCaptions
from utils.datasets.mscoco_captions import MSCOCOCaptions
from utils.metrics.metrics import CD
# from utils.multimodal_similarities import multimodal_similarities
# from utils.chart_utils import boxplot
# from utils.save_objects import save_objects
# from utils.save_data import save_data
# from utils.scatter import scatter

# from models.custom_clip import CustomCLIP
# from models.custom_align import CustomALIGN
# from models.custom_imagebind import CustomImageBind
# from models.custom_cyclip import CustomCyCLIP
from models.custom_flava import CustomFLAVA
from models.custom_albef import CustomALBEF


def create_model(name: str):
    if name == 'ALBEF':
        return CustomALBEF()
    elif name == 'FLAVA':
        return CustomFLAVA()
    else:
        raise ValueError('The selected model is not implemented yet')


def apply_model(model, dataset, root_dir):
    text_features, image_features = model.encode(dataset, batch_size=32)

    result_dir = root_dir + f'/results/embeddings/{dataset.name}/{model.name}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    torch.save(text_features, result_dir+'/text.pt')
    torch.save(image_features, result_dir+'/image.pt')


if __name__ == '__main__':

    # Parse the command-line arguments and assign values to variables
    args = parse_args()
    model_name = args.MODELNAME
    test_dataset = args.DATASET
    image_root_path = args.IMAGEROOTPATH

    # Check whether dataset and model are valid
    assert test_dataset in ['mscoco',
                            'flickr30k',
                            'amazon_products']

    assert model_name in ['CLIP_ViT-B32',
                          'CLIP_RN50',
                          'ALIGN',
                          'ImageBind',
                          'CyCLIP',
                          'FLAVA',
                          'ALBEF']

    root_dir = os.path.dirname(os.path.realpath(__file__))

    model = create_model(model_name)

    if test_dataset == 'mscoco':
        dataset = MSCOCOCaptions(root=root_dir + '/data/images/mscoco_val2017/',
						annFile=root_dir + '/data/annotations/mscoco_val2017/captions_val2017.json',
                        transform=model.transform)
        
    elif test_dataset == 'flickr30k':
        dataset = Flickr30kCaptions(root=root_dir + '/data/images/flickr30k/',
						annFile=root_dir + '/data/annotations/flickr30k/1000_random_samples.token',
                        transform=model.transform)

    else:
        raise ValueError('The selected dataset is not supported')


    apply_model(model, dataset, root_dir)

    # # Create the csv file of results
    # create_path_if_not_existant('results')
    # outfile = 'cmd.csv'
    # if not os.path.exists(os.path.join('results',outfile)):
    #     with open(os.path.join('results',outfile), 'w', encoding='UTF8') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(['tested dataset', 'model name', 'Txt-Txt', 'Img-Img', 'Img-Txt'])

    # model = model_name2model[model_name]

    # all_sim_img, all_dissim_img, all_sim_txt, all_dissim_txt, all_sim_txtimg, all_dissim_txtimg = \
    #     multimodal_similarities(model, test_dataset)

    # pos_type = ['Txt-Txt']*len(all_sim_txt)
    # pos_type.extend(['Img-Img']*len(all_sim_img))
    # pos_type.extend(['Txt-Img']*len(all_sim_txtimg))
    # all_sim = np.concatenate((all_sim_txt, all_sim_img, all_sim_txtimg))
    # pos_gobj = boxplot(all_sim, pos_type, 'Pos', 'dodgerblue')

    # neg_type = ['Txt-Txt']*len(all_dissim_txt)
    # neg_type.extend(['Img-Img']*len(all_dissim_img))
    # neg_type.extend(['Txt-Img']*len(all_dissim_txtimg))
    # all_dissim = np.concatenate((all_dissim_txt, all_dissim_img, all_dissim_txtimg))
    # neg_gobj = boxplot(all_dissim, neg_type, 'Neg', 'indianred')

    # outfolder = os.path.join('results', 'charts', model_name, test_dataset,)
    # create_path_if_not_existant(outfolder)
    # save_objects([pos_gobj, neg_gobj], outfolder, name='sim_distrib')

    # save_data([model_name]*len(np.concatenate((all_sim, all_dissim))),
    #           [test_dataset]*len(np.concatenate((all_sim, all_dissim))),
    #           np.concatenate((all_sim, all_dissim)),
    #           np.concatenate((pos_type, neg_type)),
    #           np.concatenate((['Pos']*len(all_sim), ['Neg']*len(all_dissim))),
    #           outfolder,
    #           'raw_distrib',)
    
    # for i in range(5):
    #     scatter(model, test_dataset, outfolder, image_root_path, i+1)