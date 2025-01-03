import os
import argparse

import torch
from torch.utils.data import DataLoader, SequentialSampler

from utils.datasets.flickr30k_captions import Flickr30kCaptions
from utils.datasets.mscoco_captions import MSCOCOCaptions
from utils.datasets.conceptual_captions import ConceptualCaptions

from utils.loss import compute_clip_loss

from models.custom_clip import CustomCLIP
from models.custom_align import CustomALIGN
from models.custom_imagebind import CustomImageBind
from models.custom_cyclip import CustomCyCLIP
from models.custom_flava import CustomFLAVA
from models.custom_albef import CustomALBEF
# from models.custom_visiontextdualencoder import CustomVTDE
# from models.custom_perceiver import CustomPerceiver
# from models.custom_bridgetower import CustomBridgeTower
# from models.custom_data2vec import CustomData2Vec


def create_model(name, local_path=None):
    if name == 'ALBEF':
        return CustomALBEF()
    
    elif name == 'FLAVA':
        return CustomFLAVA()
    
    elif name == 'ALIGN':
        return CustomALIGN()
    
    elif name == 'ImageBind':
        return CustomImageBind()
    
    elif name == 'CyCLIP':
        return CustomCyCLIP()
    
    elif name == 'zero_shot_CLIP_RN50':
        return CustomCLIP(
            vision_encoder='RN50',
            frozen_text_encoder=True,
            text_encoder_from_local=False,
            frozen_image_encoder=True,
            image_encoder_from_local=False,
            frozen_projection_layers=True,
            projection_layers_from_local=False)
    
    elif name == 'zero_shot_CLIP_ViT':
        return CustomCLIP(
            vision_encoder='ViT',
            frozen_text_encoder=True,
            text_encoder_from_local=False,
            frozen_image_encoder=True,
            image_encoder_from_local=False,
            frozen_projection_layers=True,
            projection_layers_from_local=False)

    elif name == 'CLIP_RN50':
        return CustomCLIP(
            vision_encoder='RN50',
            frozen_text_encoder=True,
            text_encoder_from_local=True,
            frozen_image_encoder=True,
            image_encoder_from_local=True,
            frozen_projection_layers=True,
            projection_layers_from_local=True,
            local_pretrained_weights_path=local_path)
    
    elif name == 'CLIP_ViT':
        return CustomCLIP(
            vision_encoder='ViT',
            frozen_text_encoder=True,
            text_encoder_from_local=True,
            frozen_image_encoder=True,
            image_encoder_from_local=True,
            frozen_projection_layers=True,
            projection_layers_from_local=True,
            local_pretrained_weights_path=local_path)

    # elif name == 'VTDE_LU':
    #     return CustomVTDE(
    #         frozen_text_encoder=None,
    #         frozen_image_encoder=None,
    #         pretrained_text_encoder=None,
    #         pretrained_image_encoder=None,
    #         local_pre_trained_weights='pkgs/VTDE_LU')
    
    # elif name == 'VTDE_Lu':
    #     return CustomVTDE(
    #         frozen_text_encoder=None,
    #         frozen_image_encoder=None,
    #         pretrained_text_encoder=None,
    #         pretrained_image_encoder=None,
    #         local_pre_trained_weights='pkgs/VTDE_Lu')
    
    # elif name == 'VTDE_UU':
    #     return CustomVTDE(
    #         frozen_text_encoder=None,
    #         frozen_image_encoder=None,
    #         pretrained_text_encoder=None,
    #         pretrained_image_encoder=None,
    #         local_pre_trained_weights='pkgs/VTDE_UU')

    # elif name == 'Perceiver':
    #     model = CustomPerceiver()
    #     return model

    # elif name == 'BridgeTower':
    #     return CustomBridgeTower()

    # elif name == 'Data2Vec':
    #     return CustomData2Vec()
    
    else:
        raise ValueError('The selected model is not implemented yet')
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help='name of the model', dest='MODEL')
    parser.add_argument("--local-weights", type=str, required=False,
                        help='the local path in which the pretrained weights are saved', dest='PATH')
    parser.add_argument("--dataset", type=str, required=True, help='name of the dataset', dest='DATASET')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size', dest='BATCH_SIZE')

    args = parser.parse_args()

    return args


args = parse_args()
MODEL = args.MODEL
PATH = args.PATH
DATASET = args.DATASET
BATCH_SIZE = args.BATCH_SIZE
NUM_WORKERS = 0

# for debugging
# os.environ['TORCH_HOME']='/nfs/home/morovatian/.cache/torch'
# MODEL = 'CyCLIP'
# PATH = 'weights/CLIP/ViT32_LiUi_clip_loss_mscoco.pth'
# LOSS = 'clip'
# DATASET = 'mscoco'
# BATCH_SIZE = 2
# CPI = 5 # captions per image
# SAVE_EMBDS = False
# NUM_WORKERS = 2

assert MODEL in ['ALBEF', 'FLAVA', 'ALIGN', 'ImageBind', 'BridgeTower', 'Data2Vec', 'CyCLIP',
                 'zero_shot_CLIP_RN50', 'zero_shot_CLIP_ViT',
                 'CLIP_RN50', 'CLIP_ViT',
                 'zero_shot_VTDE',
                 'VTDE',
                 'Perceiver']

if MODEL.startswith('CLIP') or MODEL == 'VTDE':
    if PATH is None:
        raise ValueError('''if you are not going to use the pre-trained weights from the Internet
                         for the CLIP or VTDE, then you must provide local pretrained weights''')
    
assert DATASET in ['mscoco', 'flickr30k', 'conceptualCaptions']

model = create_model(MODEL, PATH)

if DATASET == 'mscoco':
    test_dataset = MSCOCOCaptions(
        root='data/images/mscoco/val2017/',
        annotations_file='data/annotations/mscoco/val2017_captions.json',
        image_transform=model.transform,
        caption_transform=model.text_tokenizer,
        classified_ann_file='data/classified/mscoco_val2017.csv')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
elif DATASET == 'flickr30k':
    test_dataset = Flickr30kCaptions(
        root='data/images/flickr30k/',
        annotations_file='data/annotations/flickr30k/test.token',
        image_transform=model.transform,
        caption_transform=model.text_tokenizer,
        no_cap_per_img=CPI)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

elif DATASET == 'conceptualCaptions':
    test_dataset = ConceptualCaptions(
        root='data/images/conceptualCaptions/',
        annotations_file='data/annotations/conceptualCaptions/test.csv',
        image_transform=model.transform,
        caption_transform=model.text_tokenizer,
        no_cap_per_img=CPI)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

else:
    raise ValueError('The selected dataset is not supported')

loss_function = compute_clip_loss

retrieval_inputs = model.encode_for_retrieval(test_dataloader, loss_function)

result_dir = f'results/embeddings/visualization/{MODEL}/{DATASET}/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

torch.save(retrieval_inputs['image_embeddings'], result_dir+'image.pt')
torch.save(retrieval_inputs['text_embeddings'], result_dir+'text.pt')