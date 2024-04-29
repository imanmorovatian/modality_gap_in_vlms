import pandas as pd


def read_dataset(dataset_name):
    img_data = {}
    txt_data = {}

    if 'flickr30k' in dataset_name:
        df = pd.read_csv(
            'data/'+dataset_name+'.csv',
            names=['img_id', 'caption', 'class_id', 'class'],
            header=None)
        print(df.head())
        for key, table in df.groupby('class'):
            img_data[key] = table['img_id'].to_list()
            txt_data[key] = table['caption'].to_list()

    elif 'mscoco' in dataset_name:
        df = pd.read_csv('data/'+dataset_name+'.csv')
        for key, table in df.groupby('class'):
            img_data[key] = table['coco_url'].to_list()
            txt_data[key] = table['caption'].to_list()
    
    elif 'amazon' in dataset_name:
        df = pd.read_csv('data/'+dataset_name+'.csv')
        for key, table in df.groupby('class'):
            img_data[key] = table['imageURLHighRes'].to_list()
            txt_data[key] = table['title'].to_list()
            
    else:
        raise ValueError('Invalid dataset name!')
    
    return img_data, txt_data