from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from pkgs.ImageBind import data
from pkgs.ImageBind.models import imagebind_model
from pkgs.ImageBind.models.imagebind_model import ModalityType

from tqdm import tqdm


class CustomImageBind():
    def __init__(self,):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.to(self.device)

        self.name = 'ImageBind'

        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711))
        
        self.transform = transforms.Compose([
                transforms.Resize((256, 256), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])
        
    def encode(self, dataset, batch_size):
        dataloader = DataLoader(dataset, batch_size=batch_size)

        image_features = []
        text_features = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                images, text = batch
                
                text = text[0]
                text = {ModalityType.TEXT: data.load_and_transform_text([text], self.device)}
                text = self.model(text)[ModalityType.TEXT]

                text_features.append(text)

                images = {ModalityType.VISION: data.load_and_transform_vision_data([images], self.device)}
                images = self.model(input)[ModalityType.VISION]

                image_features.append(images)

            text_features = torch.vstack(text_features)
            text_features = torch.nn.functional.normalize(text_features, p=2.0, dim=1)

            image_features = torch.vstack(image_features)
            image_features = torch.nn.functional.normalize(image_features, p=2.0, dim=1)

        return text_features.cpu().squeeze(), image_features.cpu().squeeze()