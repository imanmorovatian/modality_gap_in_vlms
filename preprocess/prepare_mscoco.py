import pandas as pd
import numpy as np
import json


"""
This file parses the captions and instances json files related to MSCOCO dataset and 
generates a csv file with the following format:

    file_name,image_id,caption,coco_url,flickr_url,super_categories,categories

The captions file has the following format:

    info:<class 'str'> --> <class 'dict'>
    ----------------------
    licenses:<class 'str'> --> <class 'list'>
    ----------------------
    images:<class 'str'> --> <class 'list'>
    ----------------------
    annotations:<class 'str'> --> <class 'list'>
    ----------------------

The instances file has the following format:

    info:<class 'str'> --> <class 'dict'>
    ----------------------------
    licenses:<class 'str'> --> <class 'list'>
    ----------------------------
    images:<class 'str'> --> <class 'list'>
    ----------------------------
    annotations:<class 'str'> --> <class 'list'>
    ----------------------------
    categories:<class 'str'> --> <class 'list'>
    ----------------------------
"""

with open('preprocess/mscoco_json_files/captions_val2017.json') as f:
    json_file = json.load(f)
    
    captions = json_file['annotations']
    df_captions = pd.DataFrame(captions)

    images = json_file['images']
    df_images = pd.DataFrame(images)
    df_images = df_images[['file_name', 'id', 'coco_url', 'flickr_url']]

    df_img_cap = pd.merge(df_images, df_captions, left_on='id', right_on='image_id')
    df_img_cap = df_img_cap[['file_name', 'image_id', 'caption', 'coco_url', 'flickr_url']]

    df_img_cap_final = pd.DataFrame(
        [
            {
                'file_name': info['file_name'].to_list()[0],
                'image_id': img_id,
                'caption': np.random.choice(info['caption']),
                'coco_url': info['coco_url'].to_list()[0],
                'flickr_url': info['flickr_url'].to_list()[0]
            }
            for img_id, info in df_img_cap.groupby('image_id')
        ]
    )


    with open('preprocess/mscoco_json_files/instances_val2017.json') as f:
        json_file = json.load(f)

        df_img_cat = pd.DataFrame(
            [
                {
                    'image_id':ant['image_id'],
                    'category_id':ant['category_id']
                }
                for ant in json_file['annotations']
            ]
        )

        df_cat_name = pd.DataFrame(
            [
                {
                    'category_id':cat['id'],
                    'supercategory':cat['supercategory'],
                    'name':cat['name']
                }
                for cat in json_file['categories']
            ]
        )

        df_img_cat_name = pd.merge(df_img_cat, df_cat_name, how='inner', on='category_id')

        df_img_cat_final = pd.DataFrame(
            [
                {
                    'image_id': img_id,
                    'super_categories': ', '.join(set(info['supercategory'].to_list())),
                    'categories': ', '.join(set(info['name']))
                }
                for img_id, info in df_img_cat_name.groupby('image_id')
            ]
        )

        df_final = pd.merge(df_img_cap_final, df_img_cat_final, how='inner', on='image_id')
        df_final = df_final[['file_name', 'image_id', 'caption', 'coco_url', 'flickr_url', 'super_categories', 'categories']]
        
        classes = df_final['super_categories'].unique().tolist()
        class_ids = {name:idx for idx, name in enumerate(classes)}
        df_final['class'] = df_final['super_categories'].apply(lambda x: class_ids[x])
        df_final = df_final[['coco_url', 'caption', 'class', 'super_categories']]
        df_final.to_csv('data/mscoco_classified.csv', index=False)