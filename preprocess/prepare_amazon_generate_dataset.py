import pandas as pd


no_imgs_per_cat = 400

# The images of the following categories are identical to each other, so it is
# better to combine the categories

file_names = ['CDs and Vinyl.csv', 'Movies and TV.csv', 'Software.csv', 'Video Games.csv']
file_names = ['preprocess/amazon_edited_categories/'+name for name in file_names]
temp = pd.concat(map(pd.read_csv, file_names), ignore_index=True)
temp = temp[['title', 'imageURLHighRes']]
temp['class'] = 'Digital Media'
temp = temp.sample(no_imgs_per_cat)
dfs = [temp,]

categories = {
    'Appliances.csv': 'category2',
    'Arts Crafts and Sewing.csv': 'category3',
    'Automotive.csv': 'category5',
    'Books.csv': 'category0',
    # 'CDs and Vinyl.csv': 'category0',
    'Cell Phones and Accessories.csv': 'category1',
    'Clothing Shoes and Jewelry.csv': 'category3',
    'Electronics.csv': 'category1',
    'Gift Cards.csv': 'category0',
    'Grocery and Gourmet Food.csv': 'category1',
    'Home and Kitchen.csv': 'category1',
    'Industrial and Scientific.csv': 'category3',
    'Magazine Subscriptions.csv': 'category0',
    # 'Movies and TV.csv': 'category0',
    'Musical Instruments.csv': 'category1',
    'Office Products.csv': 'category2',
    'Patio Lawn and Garden.csv': 'category3',
    'Pet Supplies.csv': 'category2',
    # 'Software.csv': 'category0',
    'Sports and Outdoors.csv': 'category3',
    'Tools and Home Improvement.csv': 'category1',
    'Toys and Games.csv': 'category1'
    # 'Video Games.csv': 'category1'
}

for csv_file, hierarchy_level in categories.items():
    print(f'Wokring on {csv_file}')
    df_temp = pd.read_csv('preprocess/amazon_edited_categories/'+csv_file)
    hierarchies = dict(df_temp[hierarchy_level].value_counts())
    hierarchies = [h for h,count in hierarchies.items() if count >= no_imgs_per_cat]

    for h in hierarchies:
        print(f'\tWorking on {h}')
        df_sub_temp = df_temp[df_temp[hierarchy_level] == h].sample(no_imgs_per_cat)
        df_sub_temp['class'] = h
        df_sub_temp = df_sub_temp[['title', 'imageURLHighRes', 'class']]
        dfs.append(df_sub_temp)

df_final = pd.concat(dfs, ignore_index=True)
df_final = df_final[['imageURLHighRes', 'title', 'class']]
df_final.to_csv('data/amazon_products.csv', index=False)