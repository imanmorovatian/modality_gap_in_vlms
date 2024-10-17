import json
from collections import defaultdict
import numpy as np


with open('data/annotations/mscoco/val2017_captions.json') as f:
    data = json.load(f)
    no_test_samples = len(data['images'])


train_data = {}
test_data = {}
with open('data/annotations/mscoco/train2017full_captions.json') as f:
    data = json.load(f) # a dict with keys: info, licenses, images, annotations
    
    train_data['info'] = data['info']
    train_data['licenses'] = data['licenses']
    
    test_data['info'] = data['info']
    test_data['licenses'] = data['licenses']

    
    img_info = {}
    # a list in which each element (dict) has license, file_name, coco_url, height, width, date_captured,
    # flickr_url, (id: id of image) --> common
    for info in data['images']:
        img_info[ info['id'] ] = {
            'license': info['license'],
            'file_name': info['file_name'],
            'coco_url': info['coco_url'],
            'height': info['height'],
            'width': info['width'],
            'date_captured': info['date_captured'],
            'flickr_url': info['flickr_url']
        }

    img_cap = defaultdict(lambda: {'caption_ids': [], 'captions': []})
    # a list in which each element (dict) has image_id --> common, (id: id of caption), caption
    for info in data['annotations']:
        img_cap[ info['image_id'] ]['caption_ids'].append(info['id'])
        img_cap[ info['image_id'] ]['captions'].append(info['caption'])

    
    test_img_ids = list(map(int, np.random.choice(list(img_info.keys()), size=no_test_samples, replace=False)))
    train_img_ids = [img_id for img_id in list(img_info.keys()) if img_id not in test_img_ids]

    
    captions = []
    imgs = []
    for img_id in train_img_ids:
        img_obj = {
            'id': img_id,
            'license': img_info[img_id]['license'],
            'file_name': img_info[img_id]['file_name'],
            'coco_url': img_info[img_id]['coco_url'],
            'height': img_info[img_id]['height'],
            'width': img_info[img_id]['width'],
            'date_captured': img_info[img_id]['date_captured'],
            'flickr_url': img_info[img_id]['flickr_url'],  
        }

        imgs.append(img_obj)

        for caption_id, caption in zip(img_cap[img_id]['caption_ids'], img_cap[img_id]['captions']):
            cap_obj = {
                'image_id': img_id,
                'id': caption_id,
                'caption': caption
            }

            captions.append(cap_obj)

    train_data['images'] = imgs
    train_data['annotations'] = captions


    captions = []
    imgs = []
    for img_id in test_img_ids:
        img_obj = {
            'id': img_id,
            'license': img_info[img_id]['license'],
            'file_name': img_info[img_id]['file_name'],
            'coco_url': img_info[img_id]['coco_url'],
            'height': img_info[img_id]['height'],
            'width': img_info[img_id]['width'],
            'date_captured': img_info[img_id]['date_captured'],
            'flickr_url': img_info[img_id]['flickr_url'],  
        }

        imgs.append(img_obj)

        for caption_id, caption in zip(img_cap[img_id]['caption_ids'], img_cap[img_id]['captions']):
            cap_obj = {
                'image_id': img_id,
                'id': caption_id,
                'caption': caption
            }

            captions.append(cap_obj)

    test_data['images'] = imgs
    test_data['annotations'] = captions



with open('data/annotations/mscoco/test2017_captions.json', 'w') as fp:
    json.dump(test_data, fp)

with open('data/annotations/mscoco/train2017_captions.json', 'w') as fp:
    json.dump(train_data, fp)

    