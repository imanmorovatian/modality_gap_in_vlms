import numpy as np


with open('data/annotations/flickr30k/results_20130124.token', 'r') as f:
    image_captions = {}

    for line in f:
        img_id, cap = line.split('\t')
        img_id = img_id[:-2]
        image_captions.setdefault(img_id, [])
        image_captions[img_id].append(cap)

    images = list(image_captions.keys())

    total = len(images)
    train = (total // 5) * 3
    val = total // 5
    test = total - train - val

    all_indices = list(range(total))
    train_indices= np.random.choice(all_indices, size=train, replace=False)
    all_indices = [idx for idx in all_indices if idx not in train_indices]

    val_indices = np.random.choice(all_indices, size=val, replace=False)

    test_indices = [idx for idx in all_indices if idx not in val_indices]


    with open('data/annotations/flickr30k/train.token', 'w') as train_file:
        for idx in train_indices:
            image_id = images[idx]
            for cap_num, cap in enumerate(image_captions[image_id], start=1):
                train_file.write( f'{image_id}#{cap_num}\t{cap}' )

    with open('data/annotations/flickr30k/val.token', 'w') as val_file:
        for idx in val_indices:
            image_id = images[idx]
            for cap_num, cap in enumerate(image_captions[image_id], start=1):
                val_file.write( f'{image_id}#{cap_num}\t{cap}' )

    with open('data/annotations/flickr30k/test.token', 'w') as test_file:
        for idx in test_indices:
            image_id = images[idx]
            for cap_num, cap in enumerate(image_captions[image_id], start=1):
                test_file.write( f'{image_id}#{cap_num}\t{cap}' )

