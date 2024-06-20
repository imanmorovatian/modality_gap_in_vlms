from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import RobertaTokenizerFast, BridgeTowerImageProcessor, BridgeTowerModel

from tqdm import tqdm


class CustomBridgeTower():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.text_tokenizer = RobertaTokenizerFast.from_pretrained("FacebookAI/roberta-base")
        self.image_preprocessor = BridgeTowerImageProcessor(do_resize=False)
        self.model = BridgeTowerModel.from_pretrained('BridgeTower/bridgetower-base')
        self.model = self.model.to(self.device)
        
        self.name = 'BridgeTower'
        
        self.transform = transforms.Compose([
                transforms.Lambda(lambda img: torch.tensor(self.image_preprocessor.preprocess(img)['pixel_values']) ),
                transforms.Resize((288, 288))
            ])
    
    def encode(self, dataset, batch_size):
        dataloader = DataLoader(dataset, batch_size=batch_size)

        image_features = []
        text_features = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                images, text = batch

                text = text[0]
                text = self.text_tokenizer(text, padding=True, truncation=True, return_tensors='pt')
                text = text.to(self.device)

                images = torch.squeeze(images)
                images = images.to(self.device)

                features = self.model(input_ids=text['input_ids'], attention_mask=text['attention_mask'], pixel_values=images)

                text_features.append(features['text_features'][:,0,:])
                image_features.append(features['image_features'][:,0,:])

            text_features = torch.vstack(text_features)
            text_features = torch.nn.functional.normalize(text_features, p=2.0, dim=1)

            image_features = torch.vstack(image_features)
            image_features = torch.nn.functional.normalize(image_features, p=2.0, dim=1)

        return text_features.cpu().squeeze(), image_features.cpu().squeeze()