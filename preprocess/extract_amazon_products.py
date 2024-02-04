import json
import random
import csv
from tqdm import tqdm

'''
Only products from the clothing, shoes, and jewelry category have been selected for
inclusion in the product list. We extracted the product name, high-resolution image URL,
and product title, saving this information in the "data/amazon_products.csv" file.
'''

category = "Clothing, Shoes, and Jewellery"
products = 'Headdress, Boots, Jewellery, Wallet, Shirt, Dress, Underwear, Pants, Watch, Jacket, Sweater, Luggage'.split(', ')

def find_common_element(list1, list2):
    common_elements = [element for element in list2 if element in list1]
    if len(common_elements) == 1:
        return common_elements[0]
    return None

def read_lines_from_json_file(file_path):
    total_lines = sum(1 for line in open(file_path))
    with open(file_path, 'r') as file:
        lines = [json.loads(line) for line in tqdm(file, total=total_lines)]
    return lines

file_path = 'data/meta_Clothing_Shoes_and_Jewelry.json'
json_lines = read_lines_from_json_file(file_path)

data = []
for line in tqdm(json_lines):
    keys = line.keys()
    if 'imageURLHighRes' in keys and 'description' in keys and 'category' in keys and 'title' in keys and 1000 > len(line['title']) > 30:
        url = random.choice(line['imageURLHighRes'])
        # if not check_url(url):
        #     continue
        category = find_common_element(line['category'], products)
        if category:
            data.append([url, line['title'], category])

csv_file_path = 'data/amz-products.csv'
field_names = ['imageURL', 'title', 'category']

with open(csv_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(field_names)
    writer.writerows(data)

print(f"Data saved to {csv_file_path}")

