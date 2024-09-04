from PIL import Image

import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from pkgs.CyCLIP.clip import load as load_model


def text_tokenizer(text, *args, **kwargs):
    _, text_tokenizer = load_model(name='RN50', pretrained=False)
    return text_tokenizer.process_text(text)

class CustomCyCLIP():
    def __init__(self,):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = load_model(name='RN50', pretrained=False)
        self.model.to(self.device)

        state_dict = torch.load('pkgs/CyCLIP/cyclip_3m_best.pt', map_location = self.device)["state_dict"]
        if(next(iter(state_dict.items()))[0].startswith("module")):
            state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
        self.model.load_state_dict(state_dict)
        
        self.text_tokenizer = text_tokenizer
        self.transform = Compose([
            Resize(self.model.visual.input_resolution, interpolation = Image.BICUBIC),
            CenterCrop(self.model.visual.input_resolution),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        self.name = 'CyCLIP'
    
    def encode(self, dataloader):
        self.model.eval()
        image_features = []
        text_features = []

        with torch.no_grad():
            for batch in dataloader:
                
                images, text = batch

                text = text['input_ids']
                no_captions = text.size()[1]
                text = text.to(self.device)
                text_embeds = []

                for i in range(no_captions):
                    temp = self.model.get_text_features(text[:,i,:])
                    text_embeds.append(temp)

                text_embeds = torch.vstack(text_embeds)
                text_features.append(text_embeds)

                images = torch.squeeze(images)
                if len(images.size()) == 3:
                    images = images.unsqueeze(0)
                images = images.to(self.device)
                img_embeds = self.model.get_image_features(images)

                image_features.append(img_embeds)

            text_features = torch.vstack(text_features)
            text_features = torch.nn.functional.normalize(text_features, p=2.0, dim=1)

            image_features = torch.vstack(image_features)
            image_features = torch.nn.functional.normalize(image_features, p=2.0, dim=1)

        return text_features.cpu().squeeze(), image_features.cpu().squeeze()