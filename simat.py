import os
import pandas as pd
import argparse

import torch
import torch.nn as nn
import torchvision.datasets as datasets

from utils.metrics.simat_score import compute_simiat_scores

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
    parser.add_argument('--batch_size', type=int, required=True, help='batch size', dest='BATCH_SIZE')
    parser.add_argument('--save-embds', action='store_true', help='whether to save the embeddings of images and text', dest='SAVE_EMBDS')

    args = parser.parse_args()

    return args


args = parse_args()
MODEL = args.MODEL
PATH = args.PATH
BATCH_SIZE = args.BATCH_SIZE
SAVE_EMBDS = args.SAVE_EMBDS # whether to save the embeddings of images and text
NUM_WORKERS = 2

# for debugging
# os.environ['TORCH_HOME']='/nfs/home/morovatian/.cache/torch'
# MODEL = 'CLIP_ViT'
# PATH = 'weights/retrieval/CLIP/ViT32_LiUi_clip_loss_mscoco.pth'
# BATCH_SIZE = 32
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
    
model = create_model(MODEL, PATH)

dataset = datasets.ImageFolder('data/images/simat/', transform=model.transform)
dataloaer = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

img_enc_mapping, w2we = model.encode_for_simat(dataloaer)

if SAVE_EMBDS:
    result_dir = f'results/embeddings/simat/{MODEL}/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    torch.save(img_enc_mapping, os.path.join(result_dir, 'image.pt'))
    torch.save(w2we, os.path.join(result_dir, 'image.ptd'))

heads = torch.load('weights/simat/CLIP/head_clip_t=0.1.pt')
image_head = heads['img_head']
text_head = heads['txt_head']
# image_head = nn.Linear(512, 512)
# image_head.load_state_dict(torch.load('weights/simat/CLIP/ViT32_LlLl_clip_loss_mscoco_img_head.pth'))
# text_head = nn.Linear(512, 512)
# text_head.load_state_dict(torch.load('weights/simat/CLIP/ViT32_LlLl_clip_loss_mscoco_txt_head.pth'))

scores = compute_simiat_scores(image_head, text_head, img_enc_mapping, w2we)

result_dir = f'results/simat/{MODEL}/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

output = {
    'lambda': [],
    'score': []
}

for lambda_, score in scores.items():
    output['lambda'].append(lambda_)
    output['score'].append(score)

pd.DataFrame(output).to_csv(os.path.join(result_dir, 'simat_score.csv'), index=False)