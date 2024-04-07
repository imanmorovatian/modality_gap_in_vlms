import csv
import random
import argparse
import os

# file_path = "data/flickr30k_classified.csv"

def datapath2name(datapath:str):
    if 'flickr30k' in datapath:
        return 'flickr30k'
    if 'mscoco' in datapath:
        return 'mscoco'
    if 'amz' in datapath:
        return 'amz-products'
    raise ValueError(f"Invalid data path: {datapath}")

def save_pairs(datapath:str):
    dataname = datapath2name(datapath)

    data_dict = {}
    with open(datapath, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        if dataname == 'amz-products':
            next(csvreader)  # Skip the header row
        for row in csvreader:
            if dataname == 'amz-products':
                category = row[2]
            else:
                category = row[3].split(',')[-1].strip()
            if category not in data_dict:
                data_dict[category] = []
            imgid_or_url, caption = row[:2]
            data_dict[category].append([imgid_or_url, caption.replace('\n', ' ')])

    sorted_categories = sorted(data_dict, key=lambda x: len(data_dict[x]), reverse=True)

    n = 0
    for cat in sorted_categories:
        # print(len(data_dict[cat]))
        if len(data_dict[cat]) > 50:
            n += 1

    candidate_categories = sorted_categories[:n]

    for i in range(5):
        n_categories = 5
        sampled_categories = random.sample(candidate_categories, n_categories)

        num_samples_per_category = 50
        sampled_data = {}
        for category in sampled_categories:
            sampled_data[category] = random.sample(data_dict[category], num_samples_per_category)

        # Flatten the sampled data and write it to a CSV file
        flat_sampled_data = [(img_id, caption, category) for category, samples in sampled_data.items() for img_id, caption in samples]

        # Write to CSV
        with open(f'data/{dataname}_pairs_{i+1}.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['img id', 'caption', 'category'])
            csvwriter.writerows(flat_sampled_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process data from a the 'classified' CSV file.")
    parser.add_argument("--datapath", type=str, required=True, help="Path to the CSV file.")
    args = parser.parse_args()
    
    datapath = args.datapath
    if not os.path.isfile(datapath):
        raise ValueError(f"Invalid data path: {args.datapath}")
    save_pairs(args.datapath)