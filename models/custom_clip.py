import numpy as np
import torch
from PIL import Image

import clip

from utils.model_utils import open_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

name2encoder = {'ViT-B32' : 'ViT-B/32',
                'RN50' : 'RN50'}

class CustomCLIP():
    def __init__(self, model_name):
        
        self.vision_encoder = model_name.split('_')[-1]
        self.model, self.transform = clip.load(name2encoder[self.vision_encoder], device=DEVICE)
        self.model.to(DEVICE).eval()

        self.input_resolution = self.model.visual.input_resolution
        self.context_length = self.model.context_length
        self.vocab_size = self.model.vocab_size

        self.name = model_name
        # print("CLIP - clip parameters:", f"{np.sum([int(np.prod(p.shape)) for p in self.clip.parameters()]):,}")
    
    def encode_text(self, caption: str):
        text_tokens = clip.tokenize(caption).to(DEVICE)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.squeeze()

    def encode_image(self, image_path: str):
        rgb_pil_image = open_image(image_path).convert("RGB")
        image       = self.transform(rgb_pil_image)
        image_input = torch.tensor(np.stack([image])).to(DEVICE)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input).float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.squeeze()