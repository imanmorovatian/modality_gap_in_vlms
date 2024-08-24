import os
import argparse
from torch.utils.data import DataLoader

from utils.datasets.flickr30k_captions import Flickr30kCaptions
from utils.datasets.mscoco_captions import MSCOCOCaptions
from utils.datasets.conceptual_captions import ConceptualCaptions
from models.custom_clip import CustomCLIP


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

assert dataset in ['mscoco', 'flickr30k', 'conceptualCaptions']
    
model = CustomCLIP(pre_trained=False)

if dataset == 'mscoco':
    train_dataset = MSCOCOCaptions(root='data/images/mscoco/train2017/',
						annFile='data/annotations/mscoco/train2017edited_captions.json',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    val_dataset = MSCOCOCaptions(root='data/images/mscoco/val2017/',
						annFile='data/annotations/mscoco/val2017_captions.json',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # test split is sampled from train split
    test_dataset = MSCOCOCaptions(root='data/images/mscoco/train2017/',
						annFile='data/annotations/mscoco/test2017_captions.json',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
elif dataset == 'flickr30k':
    train_dataset = Flickr30kCaptions(root='data/images/flickr30k/',
                        annFile='data/annotations/flickr30k/train.token',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    val_dataset = Flickr30kCaptions(root='data/images/flickr30k/',
                        annFile='data/annotations/flickr30k/val.token',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    test_dataset = Flickr30kCaptions(root='data/images/flickr30k/',
                        annFile='data/annotations/flickr30k/test.token',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

elif dataset == 'conceptualCaptions':
    train_dataset = ConceptualCaptions(root='data/images/conceptualCaptions/',
                        annFile='data/annotations/conceptualCaptions/train.csv',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    val_dataset = ConceptualCaptions(root='data/images/conceptualCaptions/',
                        annFile='data/annotations/conceptualCaptions/val.csv',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    test_dataset = ConceptualCaptions(root='data/images/conceptualCaptions/',
                        annFile='data/annotations/conceptualCaptions/test.csv',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

else:
    raise ValueError('The selected dataset is not supported')


result_dir = f'pkgs/CLIP'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

model.orchestrate_training(dataset, train_dataloader, val_dataloader, test_dataloader,
                            batch_size=BATCH_SIZE, no_epochs=NO_EPOCHS, save_path=result_dir)