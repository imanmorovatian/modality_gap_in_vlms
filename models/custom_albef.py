import numpy as np
import re
import yaml
from PIL import Image

import torch
from torchvision import transforms

from utils.ALBEF.model_pretrain import ALBEF
from utils.ALBEF.tokenization_bert import BertTokenizer


class CustomALBEF:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        config = yaml.load(open('utils/ALBEF/Pretrain.yaml', 'r'), Loader=yaml.Loader)
        self.text_encoder_name = 'bert-base-uncased'

        self._tokenizer = BertTokenizer.from_pretrained(self.text_encoder_name)

        self.model = ALBEF(config=config, tokenizer=self._tokenizer, text_encoder=self.text_encoder_name)
        self.model.load_state_dict(torch.load('weights/ALBEF/ALBEF.pth',
                                              map_location=torch.device(self.device))['model'])
        self.model = self.model.to(self.device)

        self.name = 'ALBEF'

        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711))
        
        self.transform = transforms.Compose([
                transforms.Resize((256, 256), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])
    
    def _pre_caption(self, caption, max_words=30):
        caption = (
            re.sub(
                r"([,.'!?\"()*#:;~])",
                "",
                caption.lower(),
            )
            .replace("-", " ")
            .replace("/", " ")
        )

        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > max_words:
            caption = " ".join(caption_words[:max_words])
        
        return caption

    def text_tokenizer(self, text, *args, **kwargs):
        text = [self._pre_caption(t) for t in text]
        return self._tokenizer(text, **kwargs)
    
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
                
                batch_size, captions_per_image, _ = text['input_ids'].size()
                for ـ in range(batch_size):
                    # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
                    text_indices = list(range(text_index, text_index + captions_per_image))
                    image_to_text_map.append(text_indices)
                    text_index += captions_per_image

                    # Each of the next captions_per_image text captions correspond to the same image
                    text_to_image_map += [image_index] * captions_per_image
                    image_index += 1

                text['input_ids'] = torch.flatten(text['input_ids'], start_dim=0, end_dim=1)
                text['attention_mask'] = torch.flatten(text['attention_mask'], start_dim=0, end_dim=1)
                text = text.to(self.device)
                
                text_embeds = self.model.text_encoder.bert(
                     text['input_ids'],
                     attention_mask=text['attention_mask'],
                     return_dict=True,
                     mode='text'
                     ).last_hidden_state
                
                text_embeds = self.model.text_proj(text_embeds[:,0,:])
                
                images = torch.squeeze(images)
                if len(images.size()) == 3:
                    images = images.unsqueeze(0)
                images = images.to(self.device)

                img_embeds = self.model.visual_encoder(images)
                img_embeds = self.model.vision_proj(img_embeds[:,0,:])


                image_features.append(img_embeds)
                text_features.append(text_embeds)
                
                temperature = self.model.temp
                
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