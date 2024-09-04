from functools import partial

import torch
from transformers import AutoTokenizer, AutoProcessor, AlignModel


def image_transform(images, processor, *args, **kwargs):
    return processor(images=images, return_tensors='pt')


class CustomALIGN():

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AlignModel.from_pretrained("kakaobrain/align-base")
        self.model = self.model.to(self.device)
        self.text_tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")
        self.processor = AutoProcessor.from_pretrained("kakaobrain/align-base")
        self.transform = partial(image_transform, processor=self.processor)
        self.name = 'ALIGN'
	
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
                    temp = self.model.get_text_features(**input_temp).float()
                    text_embeds.append(temp)

                text_embeds = torch.vstack(text_embeds)
                text_features.append(text_embeds)

                images['pixel_values'] = torch.squeeze(images['pixel_values'] )
                if len(images['pixel_values'] .size()) == 3:
                    images['pixel_values']  = images['pixel_values'].unsqueeze(0)
                images = images.to(self.device)
                img_embeds = self.model.get_image_features(**images).float()

                image_features.append(img_embeds)

            text_features = torch.vstack(text_features)
            text_features = torch.nn.functional.normalize(text_features, p=2.0, dim=1)

            image_features = torch.vstack(image_features)
            image_features = torch.nn.functional.normalize(image_features, p=2.0, dim=1)

        return text_features.cpu().squeeze(), image_features.cpu().squeeze()