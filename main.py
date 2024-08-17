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