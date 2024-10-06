import os
import argparse
import csv

import torch
from torch.utils.data import DataLoader

from models.custom_clip import CustomCLIP
from models.custom_align import CustomALIGN
from models.custom_imagebind import CustomImageBind
from models.custom_cyclip import CustomCyCLIP
from models.custom_flava import CustomFLAVA
from models.custom_albef import CustomALBEF
from models.custom_bridgetower import CustomBridgeTower
from models.custom_data2vec import CustomData2Vec
from models.custom_perceiver import CustomPerceiver, ContrastiveLoss

from utils.datasets.flickr30k_captions import Flickr30kCaptions
from utils.datasets.mscoco_captions import MSCOCOCaptions
from utils.datasets.conceptual_captions import ConceptualCaptions

from utils.metrics.retrieval import CrossModalRetrieval
from utils.metrics.metrics import CMD, CD


def create_model(name, dataset=None):
    if name == 'ALBEF':
        return CustomALBEF()
    elif name == 'FLAVA':
        return CustomFLAVA()
    elif name == 'ALIGN':
        return CustomALIGN()
    elif name == 'ImageBind':
        return CustomImageBind()
    elif name == 'PretrainedCLIP':
        return CustomCLIP(pre_trained=True)
    elif name == 'CLIP':
        model = CustomCLIP(pre_trained=False)
        model.model.load_state_dict(torch.load(f'pkgs/CLIP/clip_{dataset}.pth'))
        return model
    elif name == 'CyCLIP':
        return CustomCyCLIP()
    elif name == 'BridgeTower':
        return CustomBridgeTower()
    elif name == 'Data2Vec':
        return CustomData2Vec()
    elif name == 'Perceiver':
        model = CustomPerceiver()
        model.model.load_state_dict(torch.load(f'pkgs/Perceiver/perceiver_{dataset}.pth'))
        return model
    else:
        raise ValueError('The selected model is not implemented yet')
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help='name of the model', dest='MODEL')
    parser.add_argument("--dataset", type=str, required=True, help='name of the dataset', dest='DATASET')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size', dest='BATCH_SIZE')
    parser.add_argument('--captions_per_image', type=int, required=True, help='number of captions per image', dest='CPI')

    args = parser.parse_args()

    return args


args = parse_args()
MODEL = args.MODEL
dataset = args.DATASET
BATCH_SIZE = args.BATCH_SIZE
CPI = args.CPI # captions per image


assert dataset in ['mscoco', 'flickr30k', 'conceptualCaptions']

assert MODEL in ['PretrainedCLIP',
                'CLIP',
                'ALIGN',
                'ImageBind',
                'CyCLIP',
                'FLAVA',
                'ALBEF',
                'BridgeTower',
                'Data2Vec',
                'Perceiver']

model = create_model(MODEL, dataset)

if dataset == 'mscoco':
    test_dataset = MSCOCOCaptions(root='data/images/mscoco/train2017/',
						annFile='data/annotations/mscoco/test2017_captions.json',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer,
                        no_cap_per_img=CPI)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
elif dataset == 'flickr30k':
    test_dataset = Flickr30kCaptions(root='data/images/flickr30k/',
                        annFile='data/annotations/flickr30k/test.token',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer,
                        no_cap_per_img=CPI)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

elif dataset == 'conceptualCaptions':
    test_dataset = ConceptualCaptions(root='data/images/conceptualCaptions/',
                        annFile='data/annotations/conceptualCaptions/test.csv',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer,
                        no_cap_per_img=CPI)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

else:
    raise ValueError('The selected dataset is not supported')

criterion = ContrastiveLoss(temperature=0.5)

retrieval_inputs = model.encode_for_retrieval(test_dataloader)

metrics = {}

retrieval_obj = CrossModalRetrieval(image_encodings=retrieval_inputs['image_embeddings'],
                                    text_encodings=retrieval_inputs['text_embeddings'],
                                    text_to_image_map=retrieval_inputs['text_to_image_mapping'],
                                    image_to_text_map=retrieval_inputs['image_to_text_mapping'],
                                    cpi=CPI,
                                    search_space='unimodal',
                                    k_vals=[1,5,10])
metrics['retrieval_unimodal'] = retrieval_obj.compute()


retrieval_obj = CrossModalRetrieval(image_encodings=retrieval_inputs['image_embeddings'],
                                    text_encodings=retrieval_inputs['text_embeddings'],
                                    text_to_image_map=retrieval_inputs['text_to_image_mapping'],
                                    image_to_text_map=retrieval_inputs['image_to_text_mapping'],
                                    cpi=CPI,
                                    search_space='multimodal',
                                    k_vals=[1,5,10])
metrics['retrieval_multimodal'] = retrieval_obj.compute()

cmd = CMD()
metrics['cmd_img_txt'] = round(
    cmd(retrieval_inputs['image_embeddings'], retrieval_inputs['text_embeddings'][::5]).item(),
    2)

cd = CD()
metrics['cd_img_txt'] = round(
    cd(retrieval_inputs['image_embeddings'], retrieval_inputs['text_embeddings'][::5]).item(),
    2)


result_dir = 'results/metrics'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

if not os.path.exists(os.path.join(result_dir, 'metrics.csv')):
    with open(os.path.join(result_dir, 'metrics.csv'), 'w', encoding='UTF8') as f:
        rows = ['dataset', 'model', 'loss', 'cmd_img_txt', 'cd_img_txt']

        for k in metrics['retrieval_unimodal'][0]:
            rows.append(f'retrieval_unimodal_t2i_k={k}')
            rows.append(f'retrieval_unimodal_i2t_k={k}')

        for k in metrics['retrieval_multimodal'][0]:
            rows.append(f'retrieval_multimodal_t2i_k={k}')
            rows.append(f'retrieval_multimodal_i2t_k={k}')

        writer = csv.writer(f)
        writer.writerow(rows)

with open(os.path.join(result_dir, 'metrics.csv'), 'a', encoding='UTF8') as f:
    rows = [dataset, MODEL, retrieval_inputs['loss'], metrics['cmd_img_txt'], metrics['cd_img_txt']]

    for i in range(len(metrics['retrieval_unimodal'][0])):
        rows.append(metrics['retrieval_unimodal'][1][i])
        rows.append(metrics['retrieval_unimodal'][2][i])

    for i in range(len(metrics['retrieval_multimodal'][0])):
        rows.append(metrics['retrieval_multimodal'][1][i])
        rows.append(metrics['retrieval_multimodal'][2][i])

    writer = csv.writer(f)
    writer.writerow(rows)