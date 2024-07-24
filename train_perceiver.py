import os
import argparse

from utils.datasets.flickr30k_captions import Flickr30kCaptions
from utils.datasets.mscoco_captions import MSCOCOCaptions
from models.custom_perceiver import CustomPerceiver


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help='name of the dataset', dest='DATASET')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size', dest='BATCH_SIZE')
    parser.add_argument('--no_epochs', type=int, required=True, help='number of epochs', dest='NO_EPOCHS')

    args = parser.parse_args()

    return args


args = parse_args()
dataset = args.DATASET
BATCH_SIZE = args.BATCH_SIZE
NO_EPOCHS = args.NO_EPOCHS

assert dataset in ['mscoco', 'flickr30k', 'amazon_products']
    
model = CustomPerceiver()

if dataset == 'mscoco':
    pass
elif dataset == 'flickr30k':
    train_dataset = Flickr30kCaptions(root='data/images/flickr30k/',
                        annFile='data/annotations/flickr30k/train.token',
                        transform=model.transform)

    val_dataset = Flickr30kCaptions(root='data/images/flickr30k/',
                        annFile='data/annotations/flickr30k/val.token',
                        transform=model.transform)

    test_dataset = Flickr30kCaptions(root='data/images/flickr30k/',
                        annFile='data/annotations/flickr30k/test.token',
                        transform=model.transform)
else:
    raise ValueError('The selected dataset is not supported')


result_dir = f'results/perceiver_checkpoint/{dataset}'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

model.orchestrate_training(dataset, train_dataset, val_dataset, test_dataset,
                            batch_size=BATCH_SIZE, no_epochs=NO_EPOCHS, save_path=result_dir)