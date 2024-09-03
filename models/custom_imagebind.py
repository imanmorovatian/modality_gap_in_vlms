from functools import partial

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from pkgs.ImageBind import data
from pkgs.ImageBind.models import imagebind_model
from pkgs.ImageBind.models.imagebind_model import ModalityType


def text_tokenizer(text, device, *args, **kwargs):
    return {ModalityType.TEXT: data.load_and_transform_text(text, device)}


class CustomImageBind():
    def __init__(self, pretrained=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = imagebind_model.imagebind_huge(pretrained=pretrained)
        self.model.to(self.device)
        self.text_tokenizer = partial(text_tokenizer, device=self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        self.name = 'ImageBind'
        
    def encode(self, dataloader):
        self.model.eval()
        image_features = []
        text_features = []

        with torch.no_grad():
            for batch in dataloader:
                images, text = batch
                text = text['text']
                
                no_captions = text.size()[1]
                text = text.to(self.device)
                text_embeds = []
                
                for i in range(no_captions):
                    temp = self.model(text[:,i,:])[ModalityType.TEXT]
                    text_embeds.append(temp)

                text_embeds = torch.vstack(text_embeds)
                text_features.append(text_embeds)

                images = torch.squeeze(images)
                if len(images.size()) == 3:
                    images = images.unsqueeze(0)
                images = images.to(self.device)
                images = {ModalityType.VISION: images}
                img_embeds = self.model(images)[ModalityType.VISION]

                image_features.append(img_embeds)

            text_features = torch.vstack(text_features)
            text_features = torch.nn.functional.normalize(text_features, p=2.0, dim=1)

            image_features = torch.vstack(image_features)
            image_features = torch.nn.functional.normalize(image_features, p=2.0, dim=1)

        return text_features.cpu().squeeze(), image_features.cpu().squeeze()