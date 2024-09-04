from functools import partial
from PIL import Image

import torch
from torchvision import transforms
from transformers import AutoImageProcessor, AutoTokenizer, FlavaModel


class CustomFLAVA():
    def __init__(self,):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained("facebook/flava-full")
        self.processor = partial(self.processor, return_tensors='pt')
        self.text_tokenizer = AutoTokenizer.from_pretrained("facebook/flava-full")
        self.model = FlavaModel.from_pretrained("facebook/flava-full")
        self.model.to(self.device)
        self.transform = transforms.Compose([
                transforms.Resize((256, 256), interpolation=Image.BICUBIC),
                transforms.ToTensor()
            ])
        
        self.name = 'FLAVA'
        
    def encode(self, dataloader):
        self.model.eval()
        image_features = []
        text_features = []

        with torch.no_grad():
            for batch in dataloader:
                images, text = batch
                
                no_captions = text['input_ids'].size()[1]
                text = text.to(self.device)
                text_embeds = []

                for i in range(no_captions):
                    input_temp = {key: value[:,i,:] for key, value in text.items()}
                    temp = self.model.text_model(**input_temp)
                    temp = temp.last_hidden_state[:,0,:]
                    temp = self.model.text_projection(temp)
                    text_embeds.append(temp)

                text_embeds = torch.vstack(text_embeds)
                text_features.append(text_embeds)

                images['pixel_values'] = torch.squeeze(images['pixel_values'] )
                if len(images['pixel_values'] .size()) == 3:
                    images['pixel_values']  = images['pixel_values'].unsqueeze(0)
                images = images.to(self.device)
                images = self.model.image_model(**images)
                images = images.last_hidden_state[:,0,:]
                img_embeds = self.model.image_projection(images)

                image_features.append(img_embeds)

            text_features = torch.vstack(text_features)
            text_features = torch.nn.functional.normalize(text_features, p=2.0, dim=1)

            image_features = torch.vstack(image_features)
            image_features = torch.nn.functional.normalize(image_features, p=2.0, dim=1)

        return text_features.cpu().squeeze(), image_features.cpu().squeeze()