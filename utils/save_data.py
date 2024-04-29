import os
import csv


def save_data(model_name, test_dataset, values, pair_modality, pair_type, outfolder, name):
    
    data = list(zip(model_name, test_dataset, pair_modality, pair_type, values))

    with open(os.path.join(outfolder, f'{name}.csv'), 'w', newline='') as csvfile:
        fieldnames = ['Model', 'Dataset', 'Pair Modality', 'Pair Type', 'Cosine Similarity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write the data
        for pair in data:
            writer.writerow({'Model': pair[0],
                             'Dataset': pair[1],
                             'Pair Modality': pair[2],
                             'Pair Type': pair[3],
                             'Cosine Similarity': pair[4]})