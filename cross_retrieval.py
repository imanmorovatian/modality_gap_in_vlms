import os
import argparse
import csv

import torch
from torch.utils.data import DataLoader, SequentialSampler

from utils.datasets.flickr30k_captions import Flickr30kCaptions
from utils.datasets.mscoco_captions import MSCOCOCaptions
from utils.datasets.conceptual_captions import ConceptualCaptions

from utils.loss import compute_clip_loss, compute_CUA_loss, compute_CUAXU_loss

from models.custom_clip import CustomCLIP
from models.custom_align import CustomALIGN
from models.custom_imagebind import CustomImageBind
# from models.custom_cyclip import CustomCyCLIP
from models.custom_flava import CustomFLAVA
from models.custom_albef import CustomALBEF
# from models.custom_bridgetower import CustomBridgeTower
# from models.custom_data2vec import CustomData2Vec
# from models.custom_perceiver import CustomPerceiver
from models.custom_visiontextdualencoder import CustomVTDE

from utils.metrics.retrieval import CrossModalRetrieval
from utils.metrics.metrics import CMD, CD


def create_model(name, local_path=None):
    if name == 'ALBEF':
        return CustomALBEF()
    elif name == 'FLAVA':
        return CustomFLAVA()
    elif name == 'ALIGN':
        return CustomALIGN()
    elif name == 'ImageBind':
        return CustomImageBind()
    # elif name == 'BridgeTower':
    #     return CustomBridgeTower()
    # elif name == 'Data2Vec':
    #     return CustomData2Vec()
    # elif name == 'CyCLIP':
    #     return CustomCyCLIP()
    
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

    elif name == 'VTDE_LU':
        return CustomVTDE(
            frozen_text_encoder=None,
            frozen_image_encoder=None,
            pretrained_text_encoder=None,
            pretrained_image_encoder=None,
            local_pre_trained_weights='pkgs/VTDE_LU')
    
    elif name == 'VTDE_Lu':
        return CustomVTDE(
            frozen_text_encoder=None,
            frozen_image_encoder=None,
            pretrained_text_encoder=None,
            pretrained_image_encoder=None,
            local_pre_trained_weights='pkgs/VTDE_Lu')
    
    elif name == 'VTDE_UU':
        return CustomVTDE(
            frozen_text_encoder=None,
            frozen_image_encoder=None,
            pretrained_text_encoder=None,
            pretrained_image_encoder=None,
            local_pre_trained_weights='pkgs/VTDE_UU')

    # elif name == 'Perceiver':
    #     model = CustomPerceiver()
    #     return model
    
    else:
        raise ValueError('The selected model is not implemented yet')
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help='name of the model', dest='MODEL')
    parser.add_argument("--local-weights", type=str, required=False,
                        help='the local path in which the pretrained weights are saved', dest='PATH')
    parser.add_argument("--loss", type=str, required=True, help='name of the loss function', dest='LOSS')
    parser.add_argument("--dataset", type=str, required=True, help='name of the dataset', dest='DATASET')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size', dest='BATCH_SIZE')
    parser.add_argument('--captions_per_image', type=int, required=True, help='number of captions per image', dest='CPI')
    parser.add_argument('--save-embds', action='store_true', help='whether to save the embeddings of images and text', dest='SAVE_EMBDS')

    args = parser.parse_args()

    return args


args = parse_args()
MODEL = args.MODEL
PATH = args.PATH
LOSS = args.LOSS
DATASET = args.DATASET
BATCH_SIZE = args.BATCH_SIZE
CPI = args.CPI # captions per image
SAVE_EMBDS = args.SAVE_EMBDS # whether to save the embeddings of images and text
NUM_WORKERS = 2

# for debugging
# os.environ['TORCH_HOME']='/nfs/home/morovatian/.cache/torch'
# MODEL = 'ImageBind'
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

assert LOSS in ['clip', 'cua', 'cuaxu']

model = create_model(MODEL, PATH)

if DATASET == 'mscoco':
    test_dataset = MSCOCOCaptions(root='data/images/mscoco/val2017/',
						annotations_file='data/annotations/mscoco/val2017_captions.json',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer,
                        no_cap_per_img=CPI)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
elif DATASET == 'flickr30k':
    test_dataset = Flickr30kCaptions(root='data/images/flickr30k/',
                        annotations_file='data/annotations/flickr30k/test.token',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer,
                        no_cap_per_img=CPI)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

elif DATASET == 'conceptualCaptions':
    test_dataset = ConceptualCaptions(root='data/images/conceptualCaptions/',
                        annotations_file='data/annotations/conceptualCaptions/test.csv',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer,
                        no_cap_per_img=CPI)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

else:
    raise ValueError('The selected dataset is not supported')


if LOSS == 'clip':
    loss_function = compute_clip_loss
elif LOSS == 'cua':
    loss_function = compute_CUA_loss
elif LOSS == 'cuaxu':
    loss_function = compute_CUAXU_loss
else:
    raise ValueError('The selected loss is not supported')

retrieval_inputs = model.encode_for_retrieval(test_dataloader, loss_function)

if SAVE_EMBDS:
    result_dir = f'results/embeddings/{MODEL}/{DATASET}/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    torch.save(retrieval_inputs['image_embeddings'], result_dir+'image.pt')
    torch.save(retrieval_inputs['text_embeddings'], result_dir+'text.pt')
    

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

file_name = os.path.join(result_dir, 'pretrained.csv')
# writing column names
if not os.path.exists(file_name):
    with open(file_name, 'w', encoding='UTF8') as f:
        rows = ['dataset', 'model', 'path', 'loss', 'cmd_img_txt', 'cd_img_txt']

        for k in metrics['retrieval_unimodal'][0]:
            rows.append(f'retrieval_unimodal_t2i_k={k}')
            rows.append(f'retrieval_unimodal_i2t_k={k}')

        for k in metrics['retrieval_multimodal'][0]:
            rows.append(f'retrieval_multimodal_t2i_k={k}')
            rows.append(f'retrieval_multimodal_i2t_k={k}')

        writer = csv.writer(f)
        writer.writerow(rows)

# writing column values
with open(file_name, 'a', encoding='UTF8') as f:
    rows = [DATASET,
            MODEL,
            PATH if PATH is not None else 'None',
            retrieval_inputs['loss'],
            metrics['cmd_img_txt'],
            metrics['cd_img_txt']]

    for i in range(len(metrics['retrieval_unimodal'][0])):
        rows.append(metrics['retrieval_unimodal'][1][i])
        rows.append(metrics['retrieval_unimodal'][2][i])

    for i in range(len(metrics['retrieval_multimodal'][0])):
        rows.append(metrics['retrieval_multimodal'][1][i])
        rows.append(metrics['retrieval_multimodal'][2][i])

    writer = csv.writer(f)
    writer.writerow(rows)