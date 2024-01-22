from typing import List
import plotly.graph_objects as go
import torch
from tqdm import tqdm
import os
import csv
import argparse
import numpy as np

from PIL import Image

from utils.chart_utils import similarities, boxplot
from utils.metrics import CMD
from models.custom_clip import CustomCLIP
from models.custom_align import CustomALIGN

def create_path_if_not_existant(path):
    if not os.path.exists(path):
        os.makedirs(path)

cmd = CMD()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_name2model = {'CLIP_ViT-B32' : CustomCLIP('CLIP_ViT-B32'),
                    'CLIP_RN50'    : CustomCLIP('CLIP_RN50'),
                    'ALIGN'        : CustomALIGN(),
                    'ALBEF'        : None,}

def write_csv(name, initial: List, values: List):
    with open(name, 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(initial + values)

def saveobjects(gobjs: List, outfolder: str, name: str):
    fig = go.Figure()
    for gobj in gobjs:
        fig.add_trace(gobj)

    fig.update_layout(
        boxmode='group',
        font=dict(size=58),
        yaxis_title = "Cosine similarity",
        autosize    = False,
        width       = 1000,
        height      = 1000,
    )

    fig.update_layout(yaxis_range=[-0.2,1])
    fig.write_image(os.path.join(outfolder, f'{name}.png'), scale=2)
    fig.write_html(os.path.join(outfolder, f'{name}.html'))

def save_data(model_name, test_dataset, values, pair_modality, pair_type, outfolder, name):
    
    data = list(zip(model_name, test_dataset, pair_modality, pair_type, values))

    with open(os.path.join(outfolder, f'{name}.csv'), 'w', newline='') as csvfile:
        fieldnames = ['Model', 'Dataset', 'Pair Modality', 'Pair Type', 'Cosine Similarity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write the data
        for pair in data:
            writer.writerow({'Model': pair[0],
                             'Dataset': pair[1],
                             'Pair Modality': pair[2],
                             'Pair Type': pair[3],
                             'Cosine Similarity': pair[4]})

def multimodal_similarities(model,
                            test_dataset,
                            image_root_path):

    img_data = {}
    txt_data = {}

    with open(os.path.join('data', f'{test_dataset}_classified.csv'), "r") as f:
        total_lines = sum(1 for _ in f)

    with open(os.path.join('data', f'{test_dataset}_classified.csv'), "r") as f:
        reader = csv.reader(f, delimiter=',')
        for row in tqdm(reader, total=total_lines, desc="Reading CSV"):
            img_id, caption, label, _ = row
            if len(caption) <= 77:
                img_data.setdefault(label, []).append(img_id) if len(img_data.get(label, [])) < 2 else None
                txt_data.setdefault(label, []).append(caption) if len(txt_data.get(label, [])) < 2 else None

    img_feats1, img_feats2 = [], []
    txt_feats1, txt_feats2 = [], []

    for pair in img_data.values():
        if len(pair) == 2:
            img1, img2 = pair
            img_feats1.append(model.encode_image(Image.open(os.path.join(image_root_path, 'images', f'{img1}.jpg')).convert('RGB')))
            img_feats2.append(model.encode_image(Image.open(os.path.join(image_root_path, 'images', f'{img2}.jpg')).convert('RGB')))

    for pair in txt_data.values():
        if len(pair) == 2:
            txt1, txt2 = pair
            txt_feats1.append(model.encode_text(txt1))
            txt_feats2.append(model.encode_text(txt2))

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


def generate(test_dataset : str,
            model_name : str,):

    assert test_dataset in ['mscoco', 'flickr30k']

    assert model_name in ['CLIP_ViT-B32',
                          'CLIP_RN50',
                          'ALIGN',
                          'ALBEF',]

    model = model_name2model[model_name]

    all_sim_img, all_dissim_img, all_sim_txt, all_dissim_txt, all_sim_txtimg, all_dissim_txtimg = multimodal_similarities(model,
                                                                                                                        test_dataset,
                                                                                                                        image_root_path,)

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

    outfolder = os.path.join('results',
                             'charts',
                             model_name,
                             test_dataset,)

    create_path_if_not_existant(outfolder)

    saveobjects([pos_gobj, neg_gobj], outfolder, name='sim_distrib')
    save_data([model_name]*len(np.concatenate((all_sim, all_dissim))),
              [test_dataset]*len(np.concatenate((all_sim, all_dissim))),
              np.concatenate((all_sim, all_dissim)),
              np.concatenate((pos_type, neg_type)),
              np.concatenate((['Pos']*len(all_sim), ['Neg']*len(all_dissim))),
              outfolder,
              'raw_distrib',)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Process model name, test dataset, and image root path.")

    # Add command-line arguments
    parser.add_argument("--modelname", type=str, default="CLIP_ViT-B32", help="Name of the model")
    parser.add_argument("--dataset", type=str, default="mscoco", help="Path to the test dataset")
    parser.add_argument("--imagerootpath", type=str, default="/nfs/datasets/MSCOCO", help="Root path of the images")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Assign values to variables
    model_name = args.modelname
    test_dataset = args.dataset
    image_root_path = args.imagerootpath

    create_path_if_not_existant('results')
    outfile = 'cmd.csv'

    if not os.path.exists(os.path.join('results',outfile)):
        with open(os.path.join('results',outfile), 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(['tested dataset', 'model name', 'Txt-Txt', 'Img-Img', 'Img-Txt'])

    generate(test_dataset = test_dataset,
            model_name = model_name,)
    