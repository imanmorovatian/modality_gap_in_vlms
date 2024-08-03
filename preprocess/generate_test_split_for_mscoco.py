import json
import pandas as pd
import random


with open('data/annotations/mscoco_val2017/val2017_captions.json') as f:
    data = json.load(f)
    no_test_samples = len(data['images'])


sampled_data = {}
with open('data/annotations/mscoco_val2017/train2017_captions.json') as f:
    data = json.load(f) # a dict with keys: info, licenses, images, annotations
    sampled_data['info'] = data['info']
    sampled_data['licenses'] = data['licenses']

    captions = data['annotations'] # is a list in which each element (dict) has image_id --> common, (id: id of caption), caption
    df_captions = pd.DataFrame(captions)

    images = data['images'] # s a list in which each element (dict) has license, file_name, coco_url, height, width, date_captured,
                                # flickr_url, (id: id of image) --> common
    df_images = pd.DataFrame(images)

    df_img_cap = pd.merge(df_images,
                          df_captions.drop_duplicates(subset='image_id'),
                          left_on='id', right_on='image_id', suffixes=('_image', '_caption'))

    samples = df_img_cap.sample(n=no_test_samples, replace=False)
    samples = samples.to_dict(orient='records')

    list_images = []
    list_annotations = []

    for s in samples:
        obj_image = {
            'license': s['license'],
            'file_name': s['file_name'],
            'coco_url': s['coco_url'],
            'height': s['height'],
            'width': s['width'],
            'date_captured': s['date_captured'],
            'flickr_url': s['flickr_url'],
            'id': s['id_image'],
        }

        list_images.append(obj_image)
        images.remove(obj_image)
        
        obj_caption = {
            'image_id': s['image_id'],
            'id': s['id_caption'],
            'caption': s['caption']
        }

        list_annotations.append(obj_caption)
        captions.remove(obj_caption)

    sampled_data['images'] = list_images
    sampled_data['annotations'] = list_annotations

    data['images'] = images
    data['annotations'] = captions


    with open('data/annotations/mscoco_val2017/test2017_captions.json', 'w') as fp:
        json.dump(sampled_data, fp)

    with open('data/annotations/mscoco_val2017/train2017edited_captions.json', 'w') as fp:
        json.dump(data, fp)

    