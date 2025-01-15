import torch.nn as nn
from torchvision import datasets
import torch
import pandas as pd
import numpy as np


def take_topk(candidates, target, top_k):
    tops = [r for r in candidates if r.item() != target]

    return tops[:top_k]


def make_normalized(x):
    return x/x.norm(dim=-1, keepdim=True)


def compute_simiat_scores(image_embeddings, words_embeddings, image_head=None, text_head=None,
                          domain='test', lbds=[1,], top_k=[1,]):
    # Adopted from https://github.com/facebookresearch/SIMAT/blob/main/eval.py

    output = {}
    transfos = pd.read_csv('data/annotations/simat/transfos.csv', index_col=0)
    triplets = pd.read_csv('data/annotations/simat/triplets.csv', index_col=0)
    did2rid = dict(zip(triplets.dataset_id, triplets.index))
    rid2did = dict(zip(triplets.index, triplets.dataset_id))
    
    transfos = transfos[transfos.is_test == (domain == 'test')]
    
    transfos_did = [rid2did[rid] for rid in transfos.region_id]
    
    img_embs_stacked = torch.stack([image_embeddings[did2rid[i]] for i in range(len(image_embeddings))]).float()
    if image_head:
        img_embs_stacked = image_head(img_embs_stacked)
    img_embs_stacked = make_normalized(img_embs_stacked)
    value_embs = torch.stack([img_embs_stacked[did] for did in transfos_did])
    
    if text_head:
        w2v = {k:make_normalized(text_head(v.float())) for k, v in words_embeddings.items()}
    else:
        w2v = {k:make_normalized(v.float()) for k, v in words_embeddings.items()}
    delta_vectors = torch.stack([w2v[x.target] - w2v[x.value] for i, x in transfos.iterrows()])
    
    oscar_scores = torch.load('data/annotations/simat/oscar_similarity_matrix.pt')
    weights = 1/np.array(transfos.norm2)**.5
    weights = weights/sum(weights)

    for lbd in lbds:
        for k in top_k:
            target_embs = value_embs + lbd*delta_vectors

            nnb = (target_embs @ img_embs_stacked.T).topk(2*k).indices
            nnb_notself = [take_topk(r, t, k) for r, t in zip(nnb, transfos_did)]

            scores = []
            for ri, tc in zip(nnb_notself, transfos.target_ids):
                temp = []
                for rii in ri:
                    if oscar_scores[rii.item(), tc] > 0.5:
                        temp.append(True)
                    else:
                        temp.append(False)

                temp = any(temp)
                if temp:
                    scores.append(1)
                else:
                    scores.append(0)
            
            
            output[(lbd, k)] = 100*np.average(scores, weights=weights)

        
    return output