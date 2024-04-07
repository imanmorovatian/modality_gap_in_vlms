import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Train a change detection model for landslide delineation')

    # name of the model
    parser.add_argument("--modelname", type=str, default="CLIP_ViT-B32", help="Name of the model", dest='MODELNAME')

    # name of the dataset
    parser.add_argument("--dataset", type=str, default="mscoco", help="Name of the test dataset", dest='DATASET')

    # path to the dataset images
    parser.add_argument("--imagerootpath", type=str, default="/nfs/datasets/MSCOCO", help="Root path of the images", dest='IMAGEROOTPATH')

    # YAML config file
    parser.add_argument('--yaml', type=str, default=None, help='YAML file containing arguments to override (default: None)', dest='YAML')

    args = parser.parse_args()  # a Namespace object containing all arguments as attributes

    # if a yaml file is provided, override arguments with the ones in the file
    if args.YAML is not None:
        with open(args.YAML, 'r') as f:
            yaml_args = yaml.safe_load(f)
        args = argparse.Namespace(**{**vars(args), **yaml_args})

    return args