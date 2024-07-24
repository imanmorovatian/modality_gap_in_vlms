import numpy as np


with open('data/annotations/flickr30k/results_20130124.token', 'r') as f:
    data = [line for line in f]

    total = len(set( [line.split('#')[0] for line in data] ))
    train = (total // 5) * 3
    val = total // 5
    test = total - train -val

    all_indices = list(range(total))
    train_indices= np.random.choice(all_indices, size=train, replace=False)
    all_indices = [idx for idx in all_indices if idx not in train_indices]

    val_indices = np.random.choice(all_indices, size=val, replace=False)

    test_indices = [idx for idx in all_indices if idx not in val_indices]


    with open('data/annotations/flickr30k/train.token', 'w') as train_file:
        for idx in train_indices:
            train_file.write(data[idx])

    with open('data/annotations/flickr30k/val.token', 'w') as val_file:
        for idx in val_indices:
            val_file.write(data[idx])

    with open('data/annotations/flickr30k/test.token', 'w') as test_file:
        for idx in test_indices:
            test_file.write(data[idx])

