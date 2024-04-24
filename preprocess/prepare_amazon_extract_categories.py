import pandas as pd
import os


def edit_dataframe(df):

    def extract_categories(hierarchy):
        table = str.maketrans("", "", "'[]")
        categories = hierarchy.translate(table).split(',')
        categories = list(map( str.strip, categories ))
        return pd.Series(categories)


    categories = df['category'].apply(lambda x: extract_categories(x))
    names = {i:f'category{i}' for i in range(categories.shape[1])}
    categories = categories.rename(names, axis='columns')

    df = df[['title', 'description', 'imageURLHighRes', 'main_category']]
    return pd.concat([df, categories], axis=1)


files = os.listdir('amazoon_categories/')

total_files = len(files)

for idx, csv_file in enumerate(files):
    print(f'Processing {csv_file} {idx+1}/{total_files}')
    df_temp = pd.read_csv('amazoon_categories/'+csv_file)
    df_temp = edit_dataframe(df_temp)
    df_temp.to_csv(f'amazoon_edited_categories/{csv_file}', index=False)