import sys
import pandas as pd
import numpy as np
import json
from urllib import request
import requests
import gzip


def get_df(url, name, total, random_selection):
    """Extracts rows with not null fields and correct urls for each category"""

    selected_keys = {'imageURLHighRes', 'title', 'description', 'category'}

    i = 0
    precessed_records = 0
    df = {}

    # Fetch the online gzip file
    response = request.urlopen(url)

    # Open file in read-only binary mode
    g = gzip.open(response, 'rb')

    for line in g:
        if np.random.rand() < random_selection:
            precessed_records += 1
            sys.stdout.write(f'\rProcessed Records: {precessed_records}/{total} -- Inserted Records: {i}')
            sys.stdout.flush()
            d = json.loads(line.decode('utf-8'))

            if selected_keys.issubset(d.keys()):
                filtered = {}

                if d['category'] != None:
                    if isinstance(d['category'], list):
                        if len(d['category']) == 0:
                            continue
                        else:
                            filtered['category'] = d['category']
                    else:
                        filtered['category'] = d['category']
                else:
                    continue

                if d['description'] != None:
                    if isinstance(d['description'], list):
                        if len(d['description']) == 0:
                            continue
                        else:
                            filtered['description'] = d['description']
                    else:
                        filtered['description'] = d['description']
                else:
                    continue

                if d['title'] != None:
                    if isinstance(d['title'], list):
                        if len(d['title']) == 0:
                            continue
                        else:
                            filtered['title'] = d['title']
                    else:
                        filtered['title'] = d['title']
                else:
                    continue

                if d['imageURLHighRes'] != None:
                    if isinstance(d['imageURLHighRes'], list):
                        if len(d['imageURLHighRes']) == 0:
                            continue
                        else:
                            image_url = np.random.choice(d['imageURLHighRes'])
                            try:
                                res = requests.head(image_url)
                                if res.status_code == 200:
                                    filtered['imageURLHighRes'] = image_url
                                else:
                                    continue
                            except requests.exceptions.RequestException as e:
                                continue
                else:
                    continue


                df[i] = filtered
                i += 1


    if len(df) > 0:
        df = pd.DataFrame.from_dict(df, orient='index')
        df.dropna(inplace=True)
        df['main_category'] = [name] * len(df)

        return (True, df)

    else:
        return (False, None)

categories = {
    'Amazon Fashion': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_AMAZON_FASHION.json.gz', 186637),
    'All Beauty': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_All_Beauty.json.gz', 32992),
    'Appliances': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Appliances.json.gz', 30459),
    'Arts Crafts and Sewing': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Arts_Crafts_and_Sewing.json.gz', 303426),
    'Automotive': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Automotive.json.gz', 932019),
    'Books': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Books.json.gz', 2935525),
    'CDs and Vinyl': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_CDs_and_Vinyl.json.gz', 544442),
    'Cell Phones and Accessories': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Cell_Phones_and_Accessories.json.gz', 590269),
    'Clothing Shoes and Jewelry': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Clothing_Shoes_and_Jewelry.json.gz', 2685059),
    'Digital Music': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Digital_Music.json.gz', 465392),
    'Electronics': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Electronics.json.gz', 786868),
    'Gift Cards': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Gift_Cards.json.gz', 1548),
    'Grocery and Gourmet Food': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Grocery_and_Gourmet_Food.json.gz', 287209),
    'Home and Kitchen': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Home_and_Kitchen.json.gz', 1301225),
    'Industrial and Scientific': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Industrial_and_Scientific.json.gz', 167524),
    'Kindle Store': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Kindle_Store.json.gz', 493859),
    'Luxury Beauty': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Luxury_Beauty.json.gz', 12308),
    'Magazine Subscriptions': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Magazine_Subscriptions.json.gz', 3493),
    'Movies and TV': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Movies_and_TV.json.gz', 203970),
    'Musical Instruments': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Musical_Instruments.json.gz', 120400),
    'Office Products': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Office_Products.json.gz', 315644),
    'Patio Lawn and Garden': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Patio_Lawn_and_Garden.json.gz', 279697),
    'Pet Supplies': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Pet_Supplies.json.gz', 206141),
    'Prime Pantry': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Prime_Pantry.json.gz', 10815),
    'Software': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Software.json.gz', 26815),
    'Sports and Outdoors': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Sports_and_Outdoors.json.gz', 962876),
    'Tools and Home Improvement': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Tools_and_Home_Improvement.json.gz', 571982),
    'Toys and Games': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Toys_and_Games.json.gz', 634414),
    'Video Games': ('https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Video_Games.json.gz', 84893)
}

idx = 1
for category, (url, no_records) in categories.items():
    print(f'fetch data from category: {category} ({idx}/{29})')
    usefulness, df = get_df(url, category, no_records, 0.1)
    print()

    if usefulness:
        df.to_csv('preprocess/amazon_categories/{category}.csv', index=False)
    else:
        print('No useful records!')

    print('------------------------------------')
    idx += 1