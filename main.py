import sys

# if it is going to be executed on Google Colab, it will be needed to add the location of the virtual environment into
# the path variable

sys.path.append(
    '/content/drive/MyDrive/Colab Notebooks/PoliTo: Thesis/modality-invariance-VLMs/VLPs_env/lib/python3.10/site-packages/'
    )

import os
import csv
# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch

from tqdm import tqdm

from utils.arg_parser import parse_args
from utils.datasets.flickr30k_captions import Flickr30kCaptions
from utils.datasets.mscoco_captions import MSCOCOCaptions
from utils.metrics.metrics import CD, CMD
from utils.chart_utils import similarities

# from utils.multimodal_similarities import multimodal_similarities
# from utils.chart_utils import boxplot
# from utils.save_objects import save_objects
# from utils.save_data import save_data
# from utils.scatter import scatter

from models.custom_clip import CustomCLIP
from models.custom_align import CustomALIGN
from models.custom_imagebind import CustomImageBind
from models.custom_cyclip import CustomCyCLIP
from models.custom_flava import CustomFLAVA
from models.custom_albef import CustomALBEF
from models.custom_bridgetower import CustomBridgeTower
from models.custom_data2vec import CustomData2Vec
from models.custom_perceiver import CustomPerceiver


def create_model(name: str):
    if name == 'ALBEF':
        return CustomALBEF()
    elif name == 'FLAVA':
        return CustomFLAVA()
    elif name == 'ALIGN':
        return CustomALIGN()
    elif name == 'ImageBind':
        return CustomImageBind()
    elif name == 'CLIP_ViT-B32':
        return CustomCLIP('CLIP_ViT-B32')
    elif name == 'CLIP_RN50':
        return CustomCLIP('CLIP_RN50')
    elif name == 'CyCLIP':
        return CustomCyCLIP()
    elif name == 'BridgeTower':
        return CustomBridgeTower()
    elif name == 'Data2Vec':
        return CustomData2Vec()
    elif name == 'Perceiver':
        return CustomPerceiver()
    else:
        raise ValueError('The selected model is not implemented yet')

def apply_model():
    # Parse the command-line arguments and assign values to variables
    args = parse_args()
    model_name = args.MODELNAME
    test_dataset = args.DATASET

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
                          'ALBEF',
                          'BridgeTower',
                          'Data2Vec',
                          'Perceiver']

    model = create_model(model_name)

    if test_dataset == 'mscoco':
        dataset = MSCOCOCaptions(root='data/images/mscoco_val2017/',
						annFile='data/annotations/mscoco_val2017/captions_val2017.json',
                        transform=model.transform)
        
    elif test_dataset == 'flickr30k':
        dataset = Flickr30kCaptions(root='data/images/flickr30k/',
						annFile='data/annotations/flickr30k/1000_random_samples.token',
                        transform=model.transform)

    else:
        raise ValueError('The selected dataset is not supported')
    

    text_features, image_features = model.encode(dataset, batch_size=32)

    result_dir = f'results/embeddings/{dataset.name}/{model.name}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    torch.save(text_features, result_dir+'/text.pt')
    torch.save(image_features, result_dir+'/image.pt')

def compute_metrics(model_names, dataset_names):
    result_dir = 'results/metrics'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if not os.path.exists(os.path.join(result_dir, 'metrics.csv')):
        with open(os.path.join(result_dir, 'metrics.csv'), 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(['dataset', 'model', 'cmd_img_txt', 'cd_img_txt'])

    for model in model_names:
        for dataset in dataset_names:
            text_embedding = torch.load(f'results/embeddings/{dataset}/{model}/text.pt')
            image_embedding = torch.load(f'results/embeddings/{dataset}/{model}/image.pt')

            cmd = CMD()
            cmd_img_txt = round(cmd(image_embedding, text_embedding).item(), 2)

            cd = CD()
            cd_img_txt = round(cd(image_embedding, text_embedding).item(), 2)

            with open(os.path.join(result_dir, 'metrics.csv'), 'a', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow([dataset, model, cmd_img_txt, cd_img_txt])

def sim_dissim_boxplot(model_names, dataset):
    result_dir = 'results/charts'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    points = []
    types = []
    models = []

    for model in tqdm(model_names, total=len(model_names)):
        text_embedding = torch.load(f'results/embeddings/{dataset}/{model}/text.pt').cpu().numpy()
        image_embedding = torch.load(f'results/embeddings/{dataset}/{model}/image.pt').cpu().numpy()

        all_sim_img_txt, all_dissim_img_txt = similarities(image_embedding @ text_embedding.T)

        all_sim_img_txt = all_sim_img_txt.tolist()
        all_dissim_img_txt = all_dissim_img_txt.tolist()

        points += all_sim_img_txt
        types += ['similarity'] * len(all_sim_img_txt)
        models += [model] * len(all_sim_img_txt)

        points += all_dissim_img_txt
        types += ['dissimilarity'] * len(all_dissim_img_txt)
        models += [model] * len(all_dissim_img_txt)


    data = {
        'point': points,
        'type': types,
        'model': models
    }

    sns.set_theme(rc={'figure.figsize':(18,15)})
    ax = sns.boxplot(data, x='model', y='point', hue='type')
    ax.set(
        title='Flickr Dataset',
        xlabel=None,
        ylabel=None,
        )
    ax.tick_params(axis='x', labelrotation=45)
    plt.savefig(result_dir+f'/{dataset}.jpg')
    print('Saved the image successfully')


if __name__ == '__main__':
    pass

    # apply_model()

    # model_names = [
    #     'CLIPViTB32',
    #     'CLIPRN50',
    #     'ALIGN',
    #     'ImageBind',
    #     'CyCLIP',
    #     'FLAVA',
    #     'ALBEF',
    #     'BridgeTower',
    #     'Data2Vec',
    #     'Perceiver'
    #     ]
    # dataset_names = ['Flickr', 'MSCOCO']

    # compute_metrics(model_names, dataset_names)

    # sim_dissim_boxplot(model_names, 'MSCOCO')

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