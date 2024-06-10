from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from pkgs.CyCLIP.clip import load as load_model

from tqdm import tqdm


class CustomCyCLIP():
    def __init__(self,):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.processor = load_model(name = 'RN50', pretrained = False)
        self.model.to(self.device)

        state_dict = torch.load('pkgs/CyCLIP/cyclip_3m_best.pt', map_location = self.device)["state_dict"]
        if(next(iter(state_dict.items()))[0].startswith("module")):
            state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.transform = Compose([
            Resize(self.model.visual.input_resolution, interpolation = Image.BICUBIC),
            CenterCrop(self.model.visual.input_resolution), ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])

        self.name = 'CyCLIP'
    
    def encode(self, dataset, batch_size):
        dataloader = DataLoader(dataset, batch_size=batch_size)

        image_features = []
        text_features = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                
                images, text = batch
                
                text = text[0]

                text = self.processor.process_text(text)
                text['input_ids'].to(self.device)
                text['attention_mask'].to(self.device)
                text = self.model.get_text_features(input_ids = text['input_ids'],attention_mask = text['attention_mask'])

                text_features.append(text)

                images = images.to(self.device)
                images = self.model.get_image_features(pixel_values = images)

                image_features.append(images)

            text_features = torch.vstack(text_features)
            text_features = torch.nn.functional.normalize(text_features, p=2.0, dim=1)

            image_features = torch.vstack(image_features)
            image_features = torch.nn.functional.normalize(image_features, p=2.0, dim=1)

        return text_features.cpu().squeeze(), image_features.cpu().squeeze()