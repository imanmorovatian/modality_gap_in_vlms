from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, AutoProcessor, AlignModel

from tqdm import tqdm


class CustomALIGN():

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.align = AlignModel.from_pretrained("kakaobrain/align-base")
        self.algin = self.align.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")
        self.processor = AutoProcessor.from_pretrained("kakaobrain/align-base")

        self.name = 'ALIGN'

        self.transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.BICUBIC),
            transforms.ToTensor()
            ])
	
    def encode(self, dataset, batch_size):
        dataloader = DataLoader(dataset, batch_size=batch_size)

        image_features = []
        text_features = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                images, text = batch
                
                text = text[0]
                text = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
                text = text.to(self.device)
                text = self.align.get_text_features(**text).float()

                text_features.append(text)

                images = self.processor(images=images, return_tensors="pt")
                images = images.to(self.device)
                images = self.align.get_image_features(**images).float()

                image_features.append(images)

            text_features = torch.vstack(text_features)
            text_features = torch.nn.functional.normalize(text_features, p=2.0, dim=1)

            image_features = torch.vstack(image_features)
            image_features = torch.nn.functional.normalize(image_features, p=2.0, dim=1)

        return text_features.cpu().squeeze(), image_features.cpu().squeeze()