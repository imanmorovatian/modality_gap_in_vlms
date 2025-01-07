# Adopted from https://github.com/facebookresearch/SIMAT/blob/main/prepare_dataset.py


import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import argparse


parser = argparse.ArgumentParser(description='prepare SIMAT dataset')
parser.add_argument('--path', type=str, help='where the Visual Genome dataset is stored')
args = parser.parse_args()

path = args.path

path_to_save_imgs = 'data/images/simat'
Path(path_to_save_imgs).mkdir(exist_ok=True)

triplets = pd.read_csv('data/annotations/simat/triplets.csv')

retrieval_db = pd.read_csv('data/annotations/simat/retrieval_db.tsv', sep='\t', index_col=0)
rid2iid = dict(zip(retrieval_db.index, retrieval_db.image_id))

for i, l in tqdm(triplets.iterrows()):
    img = Image.open(path + str(rid2iid[l.region_id])+'.jpg')
    bbox = [int(x) for x in retrieval_db.loc[l.region_id].bbox.split(',')]   
    img.crop(bbox).save(f'{path_to_save_imgs}/{l.region_id}.png')   