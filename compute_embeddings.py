import os
import torch
from torch.utils.data import DataLoader
import argparse

from models.custom_clip import CustomCLIP
from models.custom_align import CustomALIGN
from models.custom_imagebind import CustomImageBind
from models.custom_cyclip import CustomCyCLIP
from models.custom_flava import CustomFLAVA
from models.custom_albef import CustomALBEF
from models.custom_bridgetower import CustomBridgeTower
from models.custom_data2vec import CustomData2Vec
from models.custom_perceiver import CustomPerceiver

from utils.datasets.flickr30k_captions import Flickr30kCaptions
from utils.datasets.mscoco_captions import MSCOCOCaptions
from utils.datasets.conceptual_captions import ConceptualCaptions


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


text_embeddings, image_embeddings = model.encode(test_dataloader)

result_dir = f'results/embeddings/{dataset}/{model.name}'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

torch.save(text_embeddings, os.path.join(result_dir, 'text.pt'))
torch.save(image_embeddings, os.path.join(result_dir, 'image.pt'))