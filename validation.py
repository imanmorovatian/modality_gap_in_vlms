import os
import csv
import torch
import argparse

from utils.metrics.retrieval import CrossModalRetrieval
from utils.metrics.metrics import CMD, CD

from models.custom_perceiver import ContrastiveLoss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help='name of the model', dest='MODEL')
    parser.add_argument("--dataset", type=str, required=True, help='name of the dataset', dest='DATASET')
    parser.add_argument('--captions_per_image', type=int, required=True, help='number of captions per image', dest='CPI')

    args = parser.parse_args()

    return args


args = parse_args()
MODEL = args.MODEL
DATASET = args.DATASET
CPI = args.CPI # captions per image
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128

assert DATASET in ['mscoco', 'flickr30k', 'conceptualCaptions']

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


text_embeddings = torch.load(f'results/embeddings/{DATASET}/{MODEL}/text.pt')
text_embeddings = text_embeddings.to(DEVICE)

image_embeddings = torch.load(f'results/embeddings/{DATASET}/{MODEL}/image.pt')
image_embeddings = image_embeddings.to(DEVICE)


temp_text_embeddings = text_embeddings[::CPI]
no_batch = image_embeddings.size()[0] // BATCH_SIZE
loss = 0.0
criterion = ContrastiveLoss(temperature=0.5)
for i in range(0, image_embeddings.size()[0], BATCH_SIZE):
    loss += criterion(temp_text_embeddings[i:i+BATCH_SIZE, :], image_embeddings[i:i+BATCH_SIZE, :]).item()

loss /= no_batch


image_to_text_map = []
text_to_image_map = []

text_index = 0
image_index = 0

for _ in range(image_embeddings.size()[0]):
    # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
    text_indices = list(range(text_index, text_index + CPI))
    image_to_text_map.append(text_indices)
    text_index += CPI

    # Each of the next captions_per_image text captions correspond to the same image
    text_to_image_map += [image_index] * CPI
    image_index += 1

text_to_image_map = torch.LongTensor(text_to_image_map).to(DEVICE)
image_to_text_map = torch.LongTensor(image_to_text_map).to(DEVICE)


metrics = {}

retrieval_obj = CrossModalRetrieval(image_encodings=image_embeddings,
                                    text_encodings=text_embeddings,
                                    text_to_image_map=text_to_image_map,
                                    image_to_text_map=image_to_text_map,
                                    cpi=CPI,
                                    search_space='unimodal',
                                    k_vals=[1,5,10])
metrics['retrieval_unimodal'] = retrieval_obj.compute()


retrieval_obj = CrossModalRetrieval(image_encodings=image_embeddings,
                                    text_encodings=text_embeddings,
                                    text_to_image_map=text_to_image_map,
                                    image_to_text_map=image_to_text_map,
                                    cpi=CPI,
                                    search_space='multimodal',
                                    k_vals=[1,5,10])
metrics['retrieval_multimodal'] = retrieval_obj.compute()


cmd = CMD()
metrics['cmd_img_txt'] = round(cmd(image_embeddings, text_embeddings).item(), 2)

cd = CD()
metrics['cd_img_txt'] = round(cd(image_embeddings, text_embeddings).item(), 2)


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
    rows = [DATASET, MODEL, loss, metrics['cmd_img_txt'], metrics['cd_img_txt']]

    for i in range(len(metrics['retrieval_unimodal'][0])):
        rows.append(metrics['retrieval_unimodal'][1][i])
        rows.append(metrics['retrieval_unimodal'][2][i])

    for i in range(len(metrics['retrieval_multimodal'][0])):
        rows.append(metrics['retrieval_multimodal'][1][i])
        rows.append(metrics['retrieval_multimodal'][2][i])

    writer = csv.writer(f)
    writer.writerow(rows)