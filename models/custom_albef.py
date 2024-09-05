import re
import yaml
from PIL import Image
from functools import partial

import torch
from torchvision import transforms

from pkgs.ALBEF.model_pretrain import ALBEF
from pkgs.ALBEF.tokenization_bert import BertTokenizer


def pre_caption(caption, max_words=30):
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


def text_tokenizer(text, tokenizer, *args, **kwargs):
    text = [pre_caption(t) for t in text]
    return tokenizer(text, **kwargs)


class CustomALBEF:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        config = yaml.load(open('pkgs/ALBEF/Pretrain.yaml', 'r'), Loader=yaml.Loader)
        self.text_encoder = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.text_encoder)
        self.text_tokenizer = partial(text_tokenizer, tokenizer=self.tokenizer)
        self.model = ALBEF(config=config, tokenizer=self.tokenizer, text_encoder=self.text_encoder)
        self.model.load_state_dict(torch.load('pkgs/ALBEF/ALBEF.pth',
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
                    temp = self.model.text_encoder.bert(input_temp['input_ids'],
                                                   attention_mask = input_temp['attention_mask'],
                                                   return_dict = True,
                                                   mode = 'text')
                    temp = temp.last_hidden_state
                    temp = self.model.text_proj(temp[:,0,:])
                    text_embeds.append(temp)
                
                text_embeds = torch.vstack(text_embeds)
                text_features.append(text_embeds)

                images = torch.squeeze(images)
                if len(images.size()) == 3:
                    images = images.unsqueeze(0)
                images = images.to(self.device)
                images = self.model.visual_encoder(images)
                img_embeds = self.model.vision_proj(images[:,0,:])

                image_features.append(img_embeds)

            text_features = torch.vstack(text_features)
            text_features = torch.nn.functional.normalize(text_features, p=2.0, dim=1)

            image_features = torch.vstack(image_features)
            image_features = torch.nn.functional.normalize(image_features, p=2.0, dim=1)

        return text_features.cpu().squeeze(), image_features.cpu().squeeze()