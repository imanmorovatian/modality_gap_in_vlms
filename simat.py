import os
import csv
import argparse
import torch
import torchvision.datasets as datasets

from utils.create_models import create_model
from utils.metrics.simat_score import compute_simiat_scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help='name of the model', dest='MODEL')
    parser.add_argument("--local-weights", type=str, required=False,
                        help='the local path in which the pretrained weights are saved', dest='PATH')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size', dest='BATCH_SIZE')
    parser.add_argument('--n', nargs='+', default=[1], help='list of values for the number of neighbors in the retrieval', dest='N')
    parser.add_argument('--lambda', nargs='+', default=[1], help='list of values for lambda', dest='LAMBDA')
    parser.add_argument('--domain', type=str, default='test', help='domain, test or dev', dest='DOMAIN')
    parser.add_argument('--save-embds', action='store_true', help='whether to save the embeddings of images and text', dest='SAVE_EMBDS')

    args = parser.parse_args()

    return args


args = parse_args()
MODEL = args.MODEL
PATH = args.PATH
BATCH_SIZE = args.BATCH_SIZE
N = [int(n) for n in args.N]
LAMBDA = [float(l) for l in args.LAMBDA]
DOMAIN = args.DOMAIN
SAVE_EMBDS = args.SAVE_EMBDS # whether to save the embeddings of images and text
NUM_WORKERS = 2

assert MODEL in [
    'zero_shot_CLIP_RN50', 'zero_shot_CLIP_ViT',
    'CLIP_ViT',
    'zero_shot_VISTA',
    'VISTA']

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

scores = compute_simiat_scores(
    img_enc_mapping, w2we, image_head=None, text_head=None,
    domain=DOMAIN, lbds=LAMBDA, top_k=N)

output = {
    'lambda': [],
    'n': [],
    'score': []
}

for (lambda_, n), score in scores.items():
    output['lambda'].append(lambda_)
    output['n'].append(n)
    output['score'].append(score)


result_dir = f'results/simat'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

file_name = os.path.join(result_dir, 'simat_scores.csv')
# writing column names
if not os.path.exists(file_name):
    with open(file_name, 'w', encoding='UTF8') as f:
        rows = ['model', 'path', 'n', 'lambda', 'score']        
        writer = csv.writer(f)
        writer.writerow(rows)

# writing column values
with open(file_name, 'a', encoding='UTF8') as f:
    for i in range(len(output['score'])):
        rows = [
            MODEL,
            PATH if PATH is not None else 'None',
            output['n'][i],
            output['lambda'][i],
            output['score'][i]
        ]

        writer = csv.writer(f)
        writer.writerow(rows)