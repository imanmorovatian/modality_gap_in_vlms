import re
import yaml
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.model_utils import open_image

from pkgs.ALBEF.model_pretrain import ALBEF
from pkgs.ALBEF.tokenization_bert import BertTokenizer


class CustomALBEF:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        config = yaml.load(open('pkgs/ALBEF/Pretrain.yaml', 'r'), Loader=yaml.Loader)
        self.text_encoder = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.text_encoder)
        self.model = ALBEF(config=config, tokenizer=self.tokenizer, text_encoder=self.text_encoder)
        self.model.load_state_dict(torch.load('pkgs/ALBEF/ALBEF.pth',
                                              map_location=torch.device(DEVICE))['model'])
        self.model.eval()
        self.name = 'ALBEF'

        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711))
        
        self.transform = transforms.Compose([
                transforms.Resize((256, 256), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])
    
    def pre_caption(self, caption, max_words=30):
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
    
    def encode(self, dataset, batch_size):
        dataloader = DataLoader(dataset, batch_size=batch_size)

        image_features = []
        text_features = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                images, text = batch

                text = torch.cat( [self.pre_caption(t) for t in text] )
                text = self.tokenizer(text, return_tensors="pt").to(self.device)
                text = self.model.text_encoder.bert(text.input_ids,
                                                   attention_mask = text.attention_mask,
                                                   return_dict = True,
                                                   mode = 'text',)
                text = text.last_hidden_state
                text = self.model.text_proj(text_features[:,:,0,:])
                
                text_features.append(text)

                images = images.to(self.device)
                images = self.model.visual_encoder(images)
                images = self.model.vision_proj(images[:,:,0,:])

                image_features.append(images)

            text_features = torch.vstack(text_features)
            text_features = torch.nn.functional.normalize(text_features, p=2.0, dim=1)

            image_features = torch.vstack(image_features)
            image_features = torch.nn.functional.normalize(image_features, p=2.0, dim=1)

        return text_features.cpu().squeeze(), image_features.cpu().squeeze()

    def encode_text(self, caption: str):
        text                    = self.pre_caption(caption)
        text_input              = self.tokenizer(text, return_tensors="pt").to(DEVICE)
        text_output             = self.model.text_encoder.bert(text_input.input_ids,
                                                            attention_mask = text_input.attention_mask,
                                                            return_dict = True,
                                                            mode = 'text',)
        with torch.no_grad():
            text_features               = text_output.last_hidden_state
            text_features               = self.model.text_proj(text_features[:,0,:])
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.squeeze()
    
    def encode_image(self, image_path: str):
        rgb_pil_image           = open_image(image_path).convert("RGB")
        img                     = self.transform(rgb_pil_image).to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            image_features              = self.model.visual_encoder(img)
            image_features              = self.model.vision_proj(image_features[:,0,:])
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.squeeze()