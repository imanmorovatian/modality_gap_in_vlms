import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AlignModel

from utils.model_utils import open_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CustomALIGN():

    def __init__(self):
        self.align     = AlignModel.from_pretrained("kakaobrain/align-base")
        self.tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")
        self.processor = AutoProcessor.from_pretrained("kakaobrain/align-base")

        self.name = 'ALIGN'
		
    def encode_text(self, caption: str):
        text_tokens = self.tokenizer(caption, padding=True, return_tensors="pt")
        with torch.no_grad():
            text_features = self.align.get_text_features(**text_tokens).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.squeeze()

    def encode_image(self, image_path: str):
        rgb_pil_image = open_image(image_path).convert("RGB")
        image       = self.processor(images=rgb_pil_image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.align.get_image_features(**image).float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.squeeze()