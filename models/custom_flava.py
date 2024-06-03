import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, AutoTokenizer, FlavaModel

from tqdm import tqdm

from utils.model_utils import open_image


class CustomFLAVA():
    def __init__(self,):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained("facebook/flava-full")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/flava-full")
        self.model = FlavaModel.from_pretrained("facebook/flava-full")
        self.model.to(self.device)
        self.model.eval()

        self.name = 'FLAVA'

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
                
                text = self.tokenizer(text, return_tensors="pt")
                text = text.to(self.device)
                text = self.model.text_model(**text)
                text = text.last_hidden_state[:, 0, :]
                text = self.model.text_projection(text)

                text_features.append(text)

                images = open_image(images)
                images = self.processor(images, return_tensors="pt")
                images = images.to(self.device)
                images = self.model.image_model(**images)
                images = images.last_hidden_state[:, 0, :]
                images = self.model.image_projection(image_features)

                image_features.append(images)

            text_features = torch.vstack(text_features)
            text_features = torch.nn.functional.normalize(text_features, p=2.0, dim=1)

            image_features = torch.vstack(image_features)
            image_features = torch.nn.functional.normalize(image_features, p=2.0, dim=1)

        return text_features.cpu().squeeze(), image_features.cpu().squeeze()

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