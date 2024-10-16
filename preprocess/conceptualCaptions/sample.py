import os
import io
import urllib
import PIL.Image
from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
import random
import json
import wandb


wandb.init(
    entity='iman_morovatian',
    project='Thesis',
    name='conceptual captions'
    )

USER_AGENT = get_datasets_user_agent()

path_to_save_images = f'data/images/conceptualCaptions'
if not os.path.exists(path_to_save_images):
    os.makedirs(path_to_save_images)

path_to_save_annotation = 'data/annotations/conceptualCaptions'
if not os.path.exists(path_to_save_annotation):
    os.makedirs(path_to_save_annotation)


def extract_image_and_caption(urls, captions, indices, split, no_samples):

    annotations = []
    ctr = 0

    while ctr < no_samples:
        idx = random.choice(indices)
        indices.remove(idx)

        try:
            request = urllib.request.Request(
                urls[idx],
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
                image_name = f'{split}{idx}'
                image.save(os.path.join(path_to_save_images, f'{image_name}.jpg'))
                ann = {'id': image_name, 'caption': captions[idx]}
                annotations.append(ann)
                ctr += 1
                wandb.log({f'{split} counter': ctr})
        except Exception:
            continue

    
    with open(os.path.join(path_to_save_annotation, f'{split}.json'), 'w') as f:
        json.dump(annotations, f)

    return indices


dataset = load_dataset('google-research-datasets/conceptual_captions', 'unlabeled')

urls = dataset['train']['image_url']
captions = dataset['train']['caption']
indices = [i for i in range(len(dataset['train']))]

NO_TRAIN_SAMPLES = 150000
NO_VAL_SAMPLES = 15000
NO_TEST_SAMPLES = 15000

indices = extract_image_and_caption(urls, captions, indices, 'train', NO_TRAIN_SAMPLES)
indices = extract_image_and_caption(urls, captions, indices, 'val', NO_VAL_SAMPLES)
indices = extract_image_and_caption(urls, captions, indices, 'test', NO_TEST_SAMPLES)

wandb.finish()