import argparse
import os
import csv
import torch

from utils.create_models import create_model
from utils.create_dataloaders import create_dataloaders
from utils.loss import compute_clip_loss, compute_CUA_loss, compute_CUAXU_loss
from utils.metrics.measure_gap import CD, CMD
from utils.metrics.retrieval import CrossModalRetrieval


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

assert MODEL in [
    'ALBEF', 'FLAVA', 'ALIGN', 'ImageBind', 'CyCLIP',
    'zero_shot_CLIP_RN50', 'zero_shot_CLIP_ViT',
    'CLIP_ViT',
    'zero_shot_VISTA',
    'VISTA']

if MODEL.startswith('CLIP') or MODEL == 'VTDE':
    if PATH is None:
        raise ValueError('''if you are not going to use the pre-trained weights from the Internet
                         for the CLIP or VTDE, then you must provide local pretrained weights''')
    
assert DATASET in ['mscoco', 'flickr30k', 'conceptualCaptions']

assert LOSS in ['clip', 'cua', 'cuaxu']

model = create_model(MODEL, PATH)

_, _, test_dataloader = create_dataloaders(
    DATASET, model.transform, model.text_tokenizer, CPI, BATCH_SIZE, NUM_WORKERS)

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
    result_dir = f'results/embeddings/retrieval/{MODEL}/{DATASET}/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if PATH:
        name = PATH.split('ViT32_')[-1]
        name = name.split('.pth')[0]
        name += '_'
    else:
        name = 'zero_shot_'

    torch.save(retrieval_inputs['image_embeddings'], result_dir+name+'image.pt')
    torch.save(retrieval_inputs['text_embeddings'], result_dir+name+'text.pt')
    

metrics = {}

retrieval_obj = CrossModalRetrieval(
    image_encodings=retrieval_inputs['image_embeddings'],
    text_encodings=retrieval_inputs['text_embeddings'],
    text_to_image_map=retrieval_inputs['text_to_image_mapping'],
    image_to_text_map=retrieval_inputs['image_to_text_mapping'],
    cpi=CPI,
    search_space='unimodal',
    k_vals=[1,5,10])

metrics['retrieval_unimodal'] = retrieval_obj.compute()


retrieval_obj = CrossModalRetrieval(
    image_encodings=retrieval_inputs['image_embeddings'],
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


result_dir = 'results/retrieval'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

file_name = os.path.join(result_dir, 'finetune.csv')
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