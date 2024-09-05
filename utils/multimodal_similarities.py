import os
import torch
import csv
import random
import numpy as np
from itertools import combinations
from tqdm import tqdm

from utils.datasets.read_dataset import read_dataset
from utils.chart_utils import similarities
from utils.metrics.metrics import CMD


def multimodal_similarities(model, test_dataset):
    cmd = CMD()

    img_data, txt_data = read_dataset(test_dataset)
        
    img_feats1, img_feats2 = [], []
    txt_feats1, txt_feats2 = [], []
    
    for similar_images, similar_texts in tqdm(list(zip(img_data.values(), txt_data.values())), desc="Encoding Images and Texts"):
        if len(similar_images) == len(similar_texts) >= 2:
            sampled_indexes = random.sample(list(combinations(list(range(len(similar_images))), 2)), 1)
            for index1, index2 in sampled_indexes:
                img1, img2 = (similar_images[index1], similar_images[index2])
                txt1, txt2 = (similar_texts[index1], similar_texts[index2])
                if test_dataset == 'flickr30k':
                    img1 = os.path.join('data/flickr30k-images/', f'{img1}.jpg')
                    img2 = os.path.join('data/flickr30k-images/', f'{img2}.jpg')
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

    with open(os.path.join('results', 'cmd.csv'), 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow([test_dataset, model.name] + [cmd_txttxt, cmd_imgimg, cmd_imgtxt])

    return all_sim_img, all_dissim_img, all_sim_txt, all_dissim_txt, all_sim_txtimg, all_dissim_txtimg