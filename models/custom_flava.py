import torch
from PIL import Image

from transformers import AutoImageProcessor, AutoTokenizer, FlavaModel
from utils.model_utils import open_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CustomFLAVA():
    def __init__(self,):

        self.processor = AutoImageProcessor.from_pretrained("facebook/flava-full")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/flava-full")
        self.model = FlavaModel.from_pretrained("facebook/flava-full")
        self.model.to(DEVICE)
        self.model.eval()

        self.name = 'FLAVA'
    
    def encode_text(self, caption: str):
        inputs = self.tokenizer(caption, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.text_model(**inputs)
        text_features = outputs.last_hidden_state[:, 0, :]
        text_features = self.model.text_projection(text_features)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.squeeze().detach()

    def encode_image(self, image_path: str):
        rgb_pil_image = open_image(image_path)
        inputs = self.processor(rgb_pil_image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.image_model(**inputs)
        image_features = outputs.last_hidden_state[:, 0, :]
        image_features = self.model.image_projection(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.squeeze().detach()