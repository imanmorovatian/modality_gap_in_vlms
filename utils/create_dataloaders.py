from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from utils.datasets.flickr30k_captions import Flickr30kCaptions
from utils.datasets.mscoco_captions import MSCOCOCaptions
from utils.datasets.conceptual_captions import ConceptualCaptions


def create_dataloaders(
        dataset_name,
        img_transform,
        txt_transform,
        no_caps_per_img,
        batch_size,
        num_workers,
        classify_imgs=False):

    if dataset_name == 'mscoco':
        train_dataset = MSCOCOCaptions(
            root='data/images/mscoco/train2017/',
            annotations_file='data/annotations/mscoco/train2017full_captions.json',
            image_transform=img_transform,
            caption_transform=txt_transform,
            no_cap_per_img=no_caps_per_img)
        
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers)

        if classify_imgs is False:
            val_dataset = MSCOCOCaptions(
                root='data/images/mscoco/val2017/',
                annotations_file='data/annotations/mscoco/val2017_captions.json',
                image_transform=img_transform,
                caption_transform=txt_transform,
                no_cap_per_img=no_caps_per_img)
        else:
            val_dataset = MSCOCOCaptions(
                root='data/images/mscoco/val2017/',
                annotations_file='data/annotations/mscoco/val2017_captions.json',
                image_transform=img_transform,
                caption_transform=txt_transform,
                no_cap_per_img=no_caps_per_img,
                classified_ann_file='data/classified/mscoco_val2017.csv')

        val_sampler = SequentialSampler(val_dataset)
        val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size, num_workers=num_workers)
        
        test_dataloader = val_dataloader
        
    elif dataset_name == 'flickr30k':
        train_dataset = Flickr30kCaptions(
            root='data/images/flickr30k/',
            annotations_file='data/annotations/flickr30k/train.token',
            image_transform=img_transform,
            caption_transform=txt_transform,
            no_cap_per_img=no_caps_per_img)
        
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers)

        val_dataset = Flickr30kCaptions(
            root='data/images/flickr30k/',
            annotations_file='data/annotations/flickr30k/val.token',
            image_transform=img_transform,
            caption_transform=txt_transform,
            no_cap_per_img=no_caps_per_img)
        
        val_sampler = SequentialSampler(val_dataset)
        val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size, num_workers=num_workers)

        test_dataset = Flickr30kCaptions(
            root='data/images/flickr30k/',
            annotations_file='data/annotations/flickr30k/test.token',
            image_transform=img_transform,
            caption_transform=txt_transform,
            no_cap_per_img=no_caps_per_img)
        
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size, num_workers=num_workers)

    elif dataset_name == 'conceptualCaptions':
        train_dataset = ConceptualCaptions(
            root='data/images/conceptualCaptions/',
            annotations_file='data/annotations/conceptualCaptions/train.csv',
            image_transform=img_transform,
            caption_transform=txt_transform,
            no_cap_per_img=no_caps_per_img)
        
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers)

        val_dataset = ConceptualCaptions(
            root='data/images/conceptualCaptions/',
            annotations_file='data/annotations/conceptualCaptions/val.csv',
            image_transform=img_transform,
            caption_transform=txt_transform,
            no_cap_per_img=no_caps_per_img)
        
        val_sampler = SequentialSampler(val_dataset)
        val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size, num_workers=num_workers)

        test_dataset = ConceptualCaptions(
            root='data/images/conceptualCaptions/',
            annotations_file='data/annotations/conceptualCaptions/test.csv',
            image_transform=img_transform,
            caption_transform=txt_transform,
            no_cap_per_img=no_caps_per_img)
        
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size, num_workers=num_workers)

    else:
        raise ValueError('The selected dataset is not supported')
    

    return train_dataloader, val_dataloader, test_dataloader