import torch

from pkgs.ImageBind.imagebind import data
from pkgs.ImageBind.imagebind.models import imagebind_model
from pkgs.ImageBind.imagebind.models.imagebind_model import ModalityType

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import os
import requests
from urllib.parse import urlparse

def path_or_url(path_or_url):
    parsed_url = urlparse(path_or_url)
    if parsed_url.scheme and parsed_url.netloc:
        return 'url'
    else:
        return 'path'

def save_image_from_url(url, save_path):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, 'wb') as file:
            file.write(response.content)
        return save_path
    except requests.exceptions.RequestException as e:
        return None
        # raise Exception(f"Failed to download image: {e}")

def delete_image(image_path):
    try:
        os.remove(image_path)
        print(f"Image deleted: {image_path}")
    except OSError as e:
        print(f"Error deleting image: {e}")

class CustomImageBind():
    def __init__(self,):

        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.to(DEVICE)

        self.name = 'ImageBind'
    
    def encode_text(self, caption: str):
        input = {ModalityType.TEXT: data.load_and_transform_text([caption], DEVICE)}
        with torch.no_grad():
            text_features = self.model(input)[ModalityType.TEXT]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.squeeze()

    def encode_image(self, image_path_or_url: str):
        if path_or_url(image_path_or_url) == 'url':
            save_image_from_url(image_path_or_url, 'temp_image.jpg')
            image_path = 'temp_image.jpg'
            input = {ModalityType.VISION: data.load_and_transform_vision_data([image_path], DEVICE)}
            delete_image(image_path)
        else:
            input = {ModalityType.VISION: data.load_and_transform_vision_data([image_path_or_url], DEVICE)}
        with torch.no_grad():
            image_features = self.model(input)[ModalityType.VISION]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.squeeze()