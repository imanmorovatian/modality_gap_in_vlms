import os
import argparse
import torch

from utils.create_models import create_model
from utils.create_dataloaders import create_dataloaders
from utils.loss import compute_clip_loss


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

model = create_model(MODEL, PATH)

_, _, test_dataloader = create_dataloaders(
    DATASET, model.transform, model.text_tokenizer, 1, BATCH_SIZE, NUM_WORKERS, True)

loss_function = compute_clip_loss

retrieval_inputs = model.encode_for_retrieval(test_dataloader, loss_function)

result_dir = f'results/embeddings/visualization/{MODEL}/{DATASET}/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

torch.save(retrieval_inputs['image_embeddings'], result_dir+'image.pt')
torch.save(retrieval_inputs['text_embeddings'], result_dir+'text.pt')