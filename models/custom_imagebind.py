import torch

from pkgs.ImageBind.imagebind import data
from pkgs.ImageBind.imagebind.models import imagebind_model
from pkgs.ImageBind.imagebind.models.imagebind_model import ModalityType

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

    def encode_image(self, image_path: str):
        input = {ModalityType.VISION: data.load_and_transform_vision_data([image_path], DEVICE)}
        with torch.no_grad():
            image_features = self.model(input)[ModalityType.VISION]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.squeeze()