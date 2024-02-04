import torch
from PIL import Image, ImageFile

from pkgs.CyCLIP.pkgs.openai.clip import load as load_model
from utils.model_utils import open_image

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CustomCyCLIP():
    def __init__(self,):

        self.model, self.processor = load_model(name = 'RN50', pretrained = False)
        self.model.to(DEVICE)

        checkpoint = 'pkgs/CyCLIP/ckpt/best.pt'

        state_dict = torch.load(checkpoint, map_location = DEVICE)["state_dict"]
        if(next(iter(state_dict.items()))[0].startswith("module")):
            state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.name = 'CyCLIP'
    
    def encode_text(self, caption: str):
        input_text = self.processor.process_text(caption)
        input_text['input_ids'].to(DEVICE)
        input_text['attention_mask'].to(DEVICE)
        with torch.no_grad():
            text_features = self.model.get_text_features(input_ids = input_text['input_ids'],
                                                        attention_mask = input_text['attention_mask'])
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.squeeze()

    def encode_image(self, image_path: str):
        img = open_image(image_path)
        input_image = self.processor.process_image(img).to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            image_features = self.model.get_image_features(pixel_values = input_image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.squeeze()