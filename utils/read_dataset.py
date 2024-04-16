import pandas as pd


def read_dataset(dataset_name):
    img_data = {}
    txt_data = {}

    if dataset_name == 'amazon_products':
        df = pd.read_csv('data/'+dataset_name+'.csv')
        for key, table in df.groupby('class'):
            img_data[key] = table['imageURLHighRes'].to_list()
            txt_data[key] = table['title'].to_list()

    else:
        raise Exception('Not implemented yet')
    
    return img_data, txt_data