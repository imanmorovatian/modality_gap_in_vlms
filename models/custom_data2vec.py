import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, Data2VecTextModel, AutoImageProcessor, Data2VecVisionModel

from tqdm import tqdm


class CustomData2Vec():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.tokenizer = AutoTokenizer.from_pretrained('facebook/data2vec-text-base')
        self.text_model = Data2VecTextModel.from_pretrained('facebook/data2vec-text-base')
        self.text_model = self.text_model.to(self.device)
        
        self.image_processor = AutoImageProcessor.from_pretrained('facebook/data2vec-vision-base')
        self.image_model = Data2VecVisionModel.from_pretrained('facebook/data2vec-vision-base')
        self.image_model = self.image_model.to(self.device)

        self.name = 'Data2Vec'
        
        self.transform = transforms.Compose([
                transforms.Lambda(lambda img: self.image_processor(img, return_tensors='pt')['pixel_values'])
            ])
        
    def encode(self, dataset, batch_size):
        dataloader = DataLoader(dataset, batch_size=batch_size)

        image_features = []
        text_features = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                images, text = batch
                
                text = text[0]
                text = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
                text = text.to(self.device)
                outputs = self.text_model(**text)
                last_hidden_states = outputs.last_hidden_state

                text_features.append(last_hidden_states[:,0,:])

                images = images.to(self.device)
                outputs = self.image_model(**images)

                last_hidden_states = outputs.last_hidden_state

                image_features.append(last_hidden_states[:,0,:])

            text_features = torch.vstack(text_features)
            text_features = torch.nn.functional.normalize(text_features, p=2.0, dim=1)

            image_features = torch.vstack(image_features)
            image_features = torch.nn.functional.normalize(image_features, p=2.0, dim=1)

        return text_features.cpu().squeeze(), image_features.cpu().squeeze()