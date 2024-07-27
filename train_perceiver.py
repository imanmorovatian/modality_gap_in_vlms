import os
import argparse
from torch.utils.data import DataLoader

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
    train_dataloader = DataLoader(train_dataset, batch_size=1)

    val_dataset = Flickr30kCaptions(root='data/images/flickr30k/',
                        annFile='data/annotations/flickr30k/val.token',
                        transform=model.transform)
    val_dataloader = DataLoader(val_dataset, batch_size=1)

    test_dataset = Flickr30kCaptions(root='data/images/flickr30k/',
                        annFile='data/annotations/flickr30k/test.token',
                        transform=model.transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

else:
    raise ValueError('The selected dataset is not supported')


result_dir = f'pkgs/Perceiver'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

model.orchestrate_training(dataset, train_dataloader, val_dataloader, test_dataloader,
                            batch_size=BATCH_SIZE, no_epochs=NO_EPOCHS, save_path=result_dir)