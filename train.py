import argparse

from utils.create_models import create_model
from utils.create_dataloaders import create_dataloaders
from utils.loss import compute_clip_loss, compute_CUA_loss, compute_CUAXU_loss


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

assert MODEL in [
    'CLIP_ViT_LL', 'CLIP_ViT_LU', 'CLIP_ViT_UL', 'CLIP_ViT_UU', 'CLIP_ViT_UL+LU', 'CLIP_ViT_LU+UL',
    'ALIGN_LL', 'ALIGN_LU', 'ALIGN_UL', 'ALIGN_UU',
    'VISTA_LU', 'VISTA_UL', 'VISTA_UU', 'VISTA_UL+LU', 'VISTA_LU+UL']

assert DATASET in ['mscoco', 'flickr30k', 'conceptualCaptions']

assert LOSS in ['clip', 'cua', 'cuaxu']

model = create_model(MODEL, PATH)

train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
    DATASET, model.transform, model.text_tokenizer, 1, BATCH_SIZE, NUM_WORKERS)

if LOSS == 'clip':
    loss_function = compute_clip_loss
elif LOSS == 'cua':
    loss_function = compute_CUA_loss
elif LOSS == 'cuaxu':
    loss_function = compute_CUAXU_loss
else:
    raise ValueError('The selected loss is not supported')


model.orchestrate_training(
    DATASET, train_dataloader, val_dataloader, test_dataloader,
    loss_function, BATCH_SIZE, NO_EPOCHS)