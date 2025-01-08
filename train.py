import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torchvision.transforms as T

from utils.datasets.flickr30k_captions import Flickr30kCaptions
from utils.datasets.mscoco_captions import MSCOCOCaptions
from utils.datasets.conceptual_captions import ConceptualCaptions

from PIL import Image

from utils.loss import compute_clip_loss, compute_CUA_loss, compute_CUAXU_loss

from models.custom_clip import CustomCLIP
from models.custom_perceiver import CustomPerceiver
from models.custom_visiontextdualencoder import CustomVTDE


def create_model(name, local_weights=None):
    if name == 'CLIP_RN50_LL':
        # just fine tunning the projection layers
        return CustomCLIP(vision_encoder='RN50',
                          frozen_text_encoder=True,
                          frozen_image_encoder=True,
                          frozen_projection_layers=False)
    
    elif name == 'CLIP_ViT_LL':
        # just fine tunning just the projection layers
        return CustomCLIP(vision_encoder='ViT',
                          frozen_text_encoder=True,
                          frozen_image_encoder=True,
                          frozen_projection_layers=False)
    
    elif name == 'CLIP_ViT_LU':
        # Freeze the image encoder, while fine tune the text encoder
        return CustomCLIP(vision_encoder='ViT',
                          frozen_text_encoder=False,
                          frozen_image_encoder=True,
                          frozen_projection_layers=False)
    
    elif name == 'CLIP_ViT_UL':
        # Freeze the text encoder, while fine tune the image encoder
        return CustomCLIP(vision_encoder='ViT',
                          frozen_text_encoder=True,
                          frozen_image_encoder=False,
                          frozen_projection_layers=False)
    
    elif name == 'CLIP_ViT_UU':
        # Fine tune the whole model
        return CustomCLIP(vision_encoder='ViT',
                          frozen_text_encoder=False,
                          frozen_image_encoder=False,
                          frozen_projection_layers=False)
    
    elif name == 'CLIP_ViT_UL+LU':
        # Firtly, finetune the image encoder (using the local weights), and then finetune the text encoder
        return CustomCLIP(vision_encoder='ViT',
                          frozen_text_encoder=False,
                          text_encoder_from_local=False,
                          frozen_image_encoder=True,
                          image_encoder_from_local=True,
                          frozen_projection_layers=False,
                          projection_layers_from_local=False,
                          local_pretrained_weights_path=local_weights)
    
    elif name == 'CLIP_ViT_LU+UL':
        # Firtly, finetune the text encoder (using the local weights), and then finetune the image encoder
        return CustomCLIP(vision_encoder='ViT',
                          frozen_text_encoder=True,
                          text_encoder_from_local=True,
                          frozen_image_encoder=False,
                          image_encoder_from_local=False,
                          frozen_projection_layers=False,
                          projection_layers_from_local=False,
                          local_pretrained_weights_path=local_weights)

    elif name == 'CLIP_ViT_heads':
        if local_weights is None:
            model = CustomCLIP(vision_encoder='Vit')
        else:
            model = CustomCLIP(vision_encoder='ViT',
                              text_encoder_from_local=True,
                              image_encoder_from_local=True,
                              projection_layers_from_local=True,
                              frozen_projection_layers=True,
                              local_pretrained_weights_path=local_weights)
        
        model.transform = T.Compose([
            T.Resize(size=256, interpolation=Image.BICUBIC),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                        std=(0.26862954, 0.26130258, 0.27577711))])

        return model
    
    elif name == 'Perceiver':
        return CustomPerceiver()
    
    elif name == 'VTDE_LU':
        return CustomVTDE(
            frozen_text_encoder=False,
            frozen_image_encoder=True,
            pretrained_text_encoder=True,
            pretrained_image_encoder=True)
    
    elif name == 'VTDE_Lu':
        return CustomVTDE(
            frozen_text_encoder=False,
            frozen_image_encoder=True,
            pretrained_text_encoder=False,
            pretrained_image_encoder=True)
    
    elif name == 'VTDE_UU':
        return CustomVTDE(
            frozen_text_encoder=False,
            frozen_image_encoder=False,
            pretrained_text_encoder=True,
            pretrained_image_encoder=True)
    else:
        raise ValueError('The selected model is not implemented yet')
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help='name of the model', dest='MODEL')
    parser.add_argument("--local-weights", type=str, required=False,
                        help='the local path in which the pretrained weights are saved', dest='PATH')
    parser.add_argument("--loss", type=str, required=True, help='name of the loss function', dest='LOSS')
    parser.add_argument("--dataset", type=str, required=True, help='name of the dataset', dest='DATASET')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size', dest='BATCH_SIZE')
    parser.add_argument('--no_epochs', type=int, required=True, help='number of epochs', dest='NO_EPOCHS')

    args = parser.parse_args()

    return args


args = parse_args()
MODEL = args.MODEL
PATH = args.PATH
LOSS = args.LOSS
DATASET = args.DATASET
BATCH_SIZE = args.BATCH_SIZE
NO_EPOCHS = args.NO_EPOCHS
NUM_WORKERS = 2

# MODEL = 'CLIP_ViT_heads'
# PATH = 'weights/retrieval/CLIP/ViT32_LiUi_clip_loss_mscoco.pth'
# LOSS = 'clip'
# DATASET = 'mscoco'
# BATCH_SIZE = 2
# NO_EPOCHS = 1
# NUM_WORKERS = 2

assert MODEL in ['CLIP_RN50_LL',
                 'CLIP_ViT_LL', 'CLIP_ViT_LU', 'CLIP_ViT_UL', 'CLIP_ViT_UU', 'CLIP_ViT_UL+LU', 'CLIP_ViT_LU+UL',
                 'CLIP_ViT_heads',
                 'VTDE_LU', 'VTDE_Lu', 'VTDE_UU',
                 'Perceiver']

assert DATASET in ['mscoco', 'flickr30k', 'conceptualCaptions']

assert LOSS in ['clip', 'cua', 'cuaxu']

model = create_model(MODEL, PATH)
    
if DATASET == 'mscoco':
    train_dataset = MSCOCOCaptions(root='data/images/mscoco/train2017/',
						annotations_file='data/annotations/mscoco/train2017full_captions.json',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    val_dataset = MSCOCOCaptions(root='data/images/mscoco/val2017/',
						annotations_file='data/annotations/mscoco/val2017_captions.json',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # test split is sampled from train split
    # test_dataset = MSCOCOCaptions(root='data/images/mscoco/train2017/',
	# 					annotations_file='data/annotations/mscoco/test2017_captions.json',
    #                     image_transform=model.transform,
    #                     caption_transform=model.text_tokenizer)
    # test_sampler = SequentialSampler(test_dataset)
    # test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    test_dataloader = val_dataloader
    
elif DATASET == 'flickr30k':
    train_dataset = Flickr30kCaptions(root='data/images/flickr30k/',
                        annotations_file='data/annotations/flickr30k/train.token',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    val_dataset = Flickr30kCaptions(root='data/images/flickr30k/',
                        annotations_file='data/annotations/flickr30k/val.token',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    test_dataset = Flickr30kCaptions(root='data/images/flickr30k/',
                        annotations_file='data/annotations/flickr30k/test.token',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

elif DATASET == 'conceptualCaptions':
    train_dataset = ConceptualCaptions(root='data/images/conceptualCaptions/',
                        annotations_file='data/annotations/conceptualCaptions/train.csv',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    val_dataset = ConceptualCaptions(root='data/images/conceptualCaptions/',
                        annotations_file='data/annotations/conceptualCaptions/val.csv',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    test_dataset = ConceptualCaptions(root='data/images/conceptualCaptions/',
                        annotations_file='data/annotations/conceptualCaptions/test.csv',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

else:
    raise ValueError('The selected dataset is not supported')

if LOSS == 'clip':
    loss_function = compute_clip_loss
elif LOSS == 'cua':
    loss_function = compute_CUA_loss
elif LOSS == 'cuaxu':
    loss_function = compute_CUAXU_loss
else:
    raise ValueError('The selected loss is not supported')

if 'heads' in MODEL:
    model.train_heads_for_simat(
        DATASET, train_dataloader, val_dataloader, test_dataloader,
        loss_function, BATCH_SIZE, NO_EPOCHS)
else:
    model.orchestrate_training(
        DATASET, train_dataloader, val_dataloader, test_dataloader,
        loss_function, BATCH_SIZE, NO_EPOCHS)