import torch
from torch.utils.data import DataLoader
import clip

from tqdm import tqdm


name2encoder = {
    'ViT-B32' : 'ViT-B/32',
    'RN50' : 'RN50'
    }


class CustomCLIP():
    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vision_encoder = model_name.split('_')[-1]
        self.model, self.transform = clip.load(name2encoder[self.vision_encoder], device=self.device)
        self.model.to(self.device).eval()

        self.input_resolution = self.model.visual.input_resolution
        self.context_length = self.model.context_length
        self.vocab_size = self.model.vocab_size

        self.name = model_name

    def encode(self, dataset, batch_size):
        dataloader = DataLoader(dataset, batch_size=batch_size)

        image_features = []
        text_features = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                images, text = batch
                
                text = text[0]
                text = clip.tokenize(text)
                text = text.to(self.device)
                text = self.model.encode_text(text).float()

                text_features.append(text)

                images = images.to(self.device)
                images = self.model.encode_image(images).float()

                image_features.append(images)

            text_features = torch.vstack(text_features)
            text_features = torch.nn.functional.normalize(text_features, p=2.0, dim=1)

            image_features = torch.vstack(image_features)
            image_features = torch.nn.functional.normalize(image_features, p=2.0, dim=1)

        return text_features.cpu().squeeze(), image_features.cpu().squeeze()