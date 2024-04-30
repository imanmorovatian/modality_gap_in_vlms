import os
import csv
import numpy as np

from utils.chart_utils import tsne_2dplot

def scatter(model, test_dataset, outfolder, image_root_path, i):

    sampled_data = {}

    with open(f'data/pairs/{test_dataset}_pairs_{i}.csv', 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)  # Read the header
        for row in csvreader:
            imgid_or_url, caption, category = row
            if category not in sampled_data:
                sampled_data[category] = []
            sampled_data[category].append([imgid_or_url, caption])

    vectorized_data = {}
    for category, entries in sampled_data.items():
        vectorized_category_data = []
        for entry in entries:
            imgid_or_url, caption = entry[0], entry[1]
            if test_dataset == 'flickr30k':
                image_path = os.path.join('data/flickr30k-images/', f'{imgid_or_url}.jpg')
            else:
                image_path = imgid_or_url
            try:
                image_vector = model.encode_image(image_path)
                text_vector = model.encode_text(caption)
            except:
                continue
            vectorized_category_data.append([image_vector, text_vector])
        vectorized_data[category] = vectorized_category_data

    image_features_list = []
    text_features_list = []
    categories = []

    for category, entries in vectorized_data.items():
        for image_vector, text_vector in entries:
            image_features_list.append(image_vector.cpu().numpy())
            text_features_list.append(text_vector.cpu().numpy())
            categories.append(str(category))

    # Convert the lists to NumPy arrays for further processing or saving
    image_features_array = np.array(image_features_list)
    text_features_array = np.array(text_features_list)
    categories_array = np.array(categories)

    X = np.vstack((image_features_array, text_features_array))
    y = [0]*len(image_features_array) + [1]*len(text_features_array)
    categories = np.concatenate((categories_array, categories_array)).tolist()

    fig, X_embedded = tsne_2dplot(X,y, categories=categories)
    X_embedded = np.round(X_embedded, decimals=2)

    fig.update_layout(
            autosize    = False,
            width       = 1000,
            height      = 600,
            plot_bgcolor  ='rgba(0,0,0,0)',
            font        = dict(
            family      = "Calibri",
            size        = 25,)
        )

    fig.write_image(os.path.join(outfolder, f'scatter_{i}.png')) #, scale=2)
    fig.write_html(os.path.join(outfolder, f'scatter_{i}.html'))
    # Create a structured array with fields x, y, modality, category
    structured_array = np.empty(X_embedded.shape[0], dtype=[('Model', 'U50'),
                                                            ('Dataset', 'U50'),
                                                            ('x', float),
                                                            ('y', float),
                                                            ('modality', 'U5'),
                                                            ('category', 'U50'),
                                                            ])
    structured_array['Model'] = [model.name]*X_embedded.shape[0]
    structured_array['Dataset'] = [test_dataset]*X_embedded.shape[0]
    structured_array['x'] = X_embedded[:, 0]
    structured_array['y'] = X_embedded[:, 1]
    structured_array['modality'] = np.array(['text' if label == 1 else 'image' for label in y])
    structured_array['category'] = categories
    # Save the structured array to CSV
    np.savetxt(os.path.join(outfolder, f'scatter_data_{i}.csv'),
               structured_array, delimiter=',',
               fmt=['%s', '%s', '%.2f', '%.2f', '%s', '%s'],
               header=','.join(structured_array.dtype.names), comments='')