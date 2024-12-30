from PIL import Image

import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from utils.CyCLIP.clip import load as load_model


class CustomCyCLIP():
    def __init__(self,):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model, self._tokenizer = load_model(name='RN50', pretrained=False)
        state_dict = torch.load('weights/CyCLIP/cyclip_3m_best.pt', map_location = self.device)['state_dict']
        if(next(iter(state_dict.items()))[0].startswith('module')):
            state_dict = {key[len('module.'):]: value for key, value in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        
        self.transform = Compose([
            Resize(self.model.visual.input_resolution, interpolation = Image.BICUBIC),
            CenterCrop(self.model.visual.input_resolution),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        self.name = 'CyCLIP'

    def text_tokenizer(self, text, *args, **kwargs):
        return self._tokenizer.process_text(text)
    
    def encode_for_retrieval(self, dataloader, loss_function):
        self.model.eval()
        
        image_to_text_map = []
        text_to_image_map = []
        text_index = 0
        image_index = 0
        total_loss = 0
        total_batches = 0
        image_features = []
        text_features = []

        with torch.no_grad():
            for batch in dataloader:
                images, text = batch
                text = text['input_ids']

                batch_size, captions_per_image, _ = text.size()
                for ـ in range(batch_size):
                    # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
                    text_indices = list(range(text_index, text_index + captions_per_image))
                    image_to_text_map.append(text_indices)
                    text_index += captions_per_image

                    # Each of the next captions_per_image text captions correspond to the same image
                    text_to_image_map += [image_index] * captions_per_image
                    image_index += 1

                text = torch.flatten(text, start_dim=0, end_dim=1)
                text = text.to(self.device)
                
                text_embeds = self.model.get_text_features(text)
                                
                images = torch.squeeze(images)
                if len(images.size()) == 3:
                    images = images.unsqueeze(0)
                images = images.to(self.device)

                img_embeds = self.model.get_image_features(images)


                image_features.append(img_embeds)
                text_features.append(text_embeds)
                
                temperature = self.model.logit_scale.exp()
                
                loss = loss_function(img_embeds, text_embeds[::captions_per_image], temperature)
                total_loss += loss.item()
                total_batches += 1


            text_to_image_map = torch.Tensor(text_to_image_map).to(self.device)
            image_to_text_map = torch.Tensor(image_to_text_map).to(self.device)

            text_features = torch.vstack(text_features)
            text_features = torch.nn.functional.normalize(text_features, p=2.0, dim=1)

            image_features = torch.vstack(image_features)
            image_features = torch.nn.functional.normalize(image_features, p=2.0, dim=1)
            
            avg_loss = total_loss / total_batches


        return {
            'text_embeddings': text_features.squeeze(),
            'image_embeddings': image_features.squeeze(),
            'text_to_image_mapping': text_to_image_map.squeeze(),
            'image_to_text_mapping': image_to_text_map.squeeze(),
            'loss': avg_loss
        }
