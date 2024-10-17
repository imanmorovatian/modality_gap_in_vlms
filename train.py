import os
import argparse
from torch.utils.data import DataLoader

from utils.datasets.flickr30k_captions import Flickr30kCaptions
from utils.datasets.mscoco_captions import MSCOCOCaptions
from utils.datasets.conceptual_captions import ConceptualCaptions

from models.custom_clip import CustomCLIP
from models.custom_perceiver import CustomPerceiver


def create_model(name):
    if name == 'CLIP':
        return CustomCLIP(pre_trained=False)
    elif name == 'Perceiver':
        return CustomPerceiver()
    else:
        raise ValueError('The selected model is not implemented yet')
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help='name of the model', dest='MODEL')
    parser.add_argument("--dataset", type=str, required=True, help='name of the dataset', dest='DATASET')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size', dest='BATCH_SIZE')
    parser.add_argument('--no_epochs', type=int, required=True, help='number of epochs', dest='NO_EPOCHS')

    args = parser.parse_args()

    return args


args = parse_args()
MODEL = args.MODEL
dataset_name = args.DATASET
BATCH_SIZE = args.BATCH_SIZE
NO_EPOCHS = args.NO_EPOCHS
NUM_WORKERS = 2


assert dataset_name in ['mscoco', 'flickr30k', 'conceptualCaptions']
assert MODEL in ['CLIP', 'Perceiver']

model = create_model(MODEL)
    
if dataset_name == 'mscoco':
    train_dataset = MSCOCOCaptions(root='data/images/mscoco/train2017/',
						annotations_file='data/annotations/mscoco/train2017_captions.json',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    val_dataset = MSCOCOCaptions(root='data/images/mscoco/val2017/',
						annotations_file='data/annotations/mscoco/val2017_captions.json',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # test split is sampled from train split
    test_dataset = MSCOCOCaptions(root='data/images/mscoco/train2017/',
						annotations_file='data/annotations/mscoco/test2017_captions.json',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
elif dataset_name == 'flickr30k':
    train_dataset = Flickr30kCaptions(root='data/images/flickr30k/',
                        annotations_file='data/annotations/flickr30k/train.token',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    val_dataset = Flickr30kCaptions(root='data/images/flickr30k/',
                        annotations_file='data/annotations/flickr30k/val.token',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    test_dataset = Flickr30kCaptions(root='data/images/flickr30k/',
                        annotations_file='data/annotations/flickr30k/test.token',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

elif dataset_name == 'conceptualCaptions':
    train_dataset = ConceptualCaptions(root='data/images/conceptualCaptions/',
                        annotations_file='data/annotations/conceptualCaptions/train.csv',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    val_dataset = ConceptualCaptions(root='data/images/conceptualCaptions/',
                        annotations_file='data/annotations/conceptualCaptions/val.csv',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    test_dataset = ConceptualCaptions(root='data/images/conceptualCaptions/',
                        annotations_file='data/annotations/conceptualCaptions/test.csv',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

else:
    raise ValueError('The selected dataset is not supported')


result_dir = f'pkgs/{MODEL}'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

model.orchestrate_training(dataset_name, val_dataloader, val_dataloader, test_dataloader,
                            BATCH_SIZE,NO_EPOCHS, result_dir)