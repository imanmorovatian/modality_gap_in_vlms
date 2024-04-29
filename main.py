import sys

# if it is going to be executed on Google Colab, it will be needed to add the location of the virtual environment into
# the path variable

sys.path.append(
    '/content/drive/MyDrive/Colab Notebooks/PoliTo: Thesis/modality-invariance-VLMs/VLPs_env/lib/python3.10/site-packages/'
    )

import os
import numpy as np
import torch
import csv
import random
from itertools import combinations
from typing import List
from tqdm import tqdm

from utils.arg_parser import parse_args
from utils.save_objects import save_objects
from utils.save_data import save_data
from utils.chart_utils import similarities, boxplot, tsne_2dplot
from utils.metrics import CMD
from utils.read_dataset import read_dataset

from models.custom_clip import CustomCLIP
from models.custom_align import CustomALIGN
from models.custom_imagebind import CustomImageBind
from models.custom_cyclip import CustomCyCLIP
from models.custom_flava import CustomFLAVA
from models.custom_albef import CustomALBEF


def create_path_if_not_existant(path):
    if not os.path.exists(path):
        os.makedirs(path)

cmd = CMD()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_name2model = {
    # 'CLIP_ViT-B32' : CustomCLIP('CLIP_ViT-B32'),
    # 'CLIP_RN50'    : CustomCLIP('CLIP_RN50'),
    # 'ALIGN'        : CustomALIGN(),
    # 'ImageBind'    : CustomImageBind(),
    # 'CyCLIP'       : CustomCyCLIP(),
    # 'FLAVA'        : CustomFLAVA(),
    'ALBEF'        : CustomALBEF()
    }

def write_csv(name, initial: List, values: List):
    with open(name, 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(initial + values)

def multimodal_similarities(model, test_dataset, image_root_path):

    img_data, txt_data = read_dataset(test_dataset)
        
    img_feats1, img_feats2 = [], []
    txt_feats1, txt_feats2 = [], []
    
    for similar_images, similar_texts in tqdm(list(zip(img_data.values(), txt_data.values())), desc="Encoding Images and Texts"):
        if len(similar_images) == len(similar_texts) >= 2:
            sampled_indexes = random.sample(list(combinations(list(range(len(similar_images))), 2)), 1)
            for index1, index2 in sampled_indexes:
                img1, img2 = (similar_images[index1], similar_images[index2])
                txt1, txt2 = (similar_texts[index1], similar_texts[index2])
                if test_dataset in ['mscoco', 'flickr30k']:
                    img1 = os.path.join(image_root_path, 'images', f'{img1}.jpg')
                    img2 = os.path.join(image_root_path, 'images', f'{img2}.jpg')
                try:
                    img_feats1.append(model.encode_image(img1))
                    img_feats2.append(model.encode_image(img2))
                    txt_feats1.append(model.encode_text(txt1))
                    txt_feats2.append(model.encode_text(txt2))
                except:
                    continue
                    
    img_feats1, img_feats2 = torch.vstack(img_feats1), torch.vstack(img_feats2)
    txt_feats1, txt_feats2 = torch.vstack(txt_feats1), torch.vstack(txt_feats2)

    all_sim_img, all_dissim_img = similarities(img_feats1.cpu().numpy() @ img_feats2.cpu().numpy().T)
    all_sim_txt, all_dissim_txt = similarities(txt_feats1.cpu().numpy() @ txt_feats2.cpu().numpy().T)
    all_sim_txtimg, all_dissim_txtimg = similarities(img_feats1.cpu().numpy() @ txt_feats1.cpu().numpy().T)

    all_dissim_img = np.random.choice(all_dissim_img, size=len(all_sim_img), replace=False)
    all_dissim_txt = np.random.choice(all_dissim_txt, size=len(all_sim_txt), replace=False)
    all_dissim_txtimg = np.random.choice(all_dissim_txtimg, size=len(all_sim_txtimg), replace=False)

    cmd_txttxt = round(cmd(txt_feats1, txt_feats2).item(), 2)
    cmd_imgimg = round(cmd(img_feats1, img_feats2).item(), 2)
    cmd_imgtxt = round(cmd(img_feats1, txt_feats1).item(), 2)

    write_csv(os.path.join('results', 'cmd.csv'), [test_dataset, model.name], [cmd_txttxt, cmd_imgimg, cmd_imgtxt])

    return all_sim_img, all_dissim_img, all_sim_txt, all_dissim_txt, all_sim_txtimg, all_dissim_txtimg

def scatter(model,
            test_dataset,
            outfolder,
            i,):

    sampled_data = {}
    with open(f'data/{test_dataset}_pairs_{i}.csv', 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)  # Read the header
        for row in csvreader:
            imgid_or_url, caption, category = row
            if category not in sampled_data:
                sampled_data[category] = []
            sampled_data[category].append([imgid_or_url, caption])

    vectorized_data = {}
    for category, entries in sampled_data.items():
        vectorized_category_data = []
        for entry in entries:
            imgid_or_url, caption = entry[0], entry[1]
            if test_dataset == 'amazon_products':
                image_path = imgid_or_url
            else:
                image_path = os.path.join(image_root_path, 'images', f'{imgid_or_url}.jpg')
            try:
                image_vector = model.encode_image(image_path)
                text_vector = model.encode_text(caption)
            except:
                continue
            vectorized_category_data.append([image_vector, text_vector])
        vectorized_data[category] = vectorized_category_data

    image_features_list = []
    text_features_list = []
    categories = []

    for category, entries in vectorized_data.items():
        for image_vector, text_vector in entries:
            image_features_list.append(image_vector.cpu().numpy())
            text_features_list.append(text_vector.cpu().numpy())
            categories.append(str(category))

    # Convert the lists to NumPy arrays for further processing or saving
    image_features_array = np.array(image_features_list)
    text_features_array = np.array(text_features_list)
    categories_array = np.array(categories)

    X = np.vstack((image_features_array, text_features_array))
    y = [0]*len(image_features_array) + [1]*len(text_features_array)
    categories = np.concatenate((categories_array, categories_array)).tolist()

    fig, X_embedded = tsne_2dplot(X,y, categories=categories)
    X_embedded = np.round(X_embedded, decimals=2)

    fig.update_layout(
            autosize    = False,
            width       = 1000,
            height      = 600,
            plot_bgcolor  ='rgba(0,0,0,0)',
            font        = dict(
            family      = "Calibri",
            size        = 25,)
        )

    fig.write_image(os.path.join(outfolder, f'scatter_{i}.png')) #, scale=2)
    fig.write_html(os.path.join(outfolder, f'scatter_{i}.html'))
    # Create a structured array with fields x, y, modality, category
    structured_array = np.empty(X_embedded.shape[0], dtype=[('Model', 'U50'),
                                                            ('Dataset', 'U50'),
                                                            ('x', float),
                                                            ('y', float),
                                                            ('modality', 'U5'),
                                                            ('category', 'U50'),
                                                            ])
    structured_array['Model'] = [model.name]*X_embedded.shape[0]
    structured_array['Dataset'] = [test_dataset]*X_embedded.shape[0]
    structured_array['x'] = X_embedded[:, 0]
    structured_array['y'] = X_embedded[:, 1]
    structured_array['modality'] = np.array(['text' if label == 1 else 'image' for label in y])
    structured_array['category'] = categories
    # Save the structured array to CSV
    np.savetxt(os.path.join(outfolder, f'scatter_data_{i}.csv'),
               structured_array, delimiter=',',
               fmt=['%s', '%s', '%.2f', '%.2f', '%s', '%s'],
               header=','.join(structured_array.dtype.names), comments='')


if __name__ == '__main__':

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
        scatter(model, test_dataset, outfolder, i+1,)