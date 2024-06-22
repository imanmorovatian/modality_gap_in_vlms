import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import PerceiverConfig, PerceiverTokenizer, PerceiverImageProcessor, PerceiverModel
from transformers.models.perceiver.modeling_perceiver import PerceiverTextPreprocessor, PerceiverImagePreprocessor

from tqdm import tqdm


class CustomPerceiver():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.text_config = PerceiverConfig.from_pretrained('deepmind/language-perceiver')
        self.text_tokenizer = PerceiverTokenizer()
        self.text_preprocessor = PerceiverTextPreprocessor(self.text_config)
        self.text_model = PerceiverModel(self.text_config, input_preprocessor=self.text_preprocessor)
        self.text_model = self.text_model.to(self.device)

        self.image_config = PerceiverConfig.from_pretrained('deepmind/vision-perceiver-learned')
        self.image_processor = PerceiverImageProcessor()
        self.image_preprocessor = PerceiverImagePreprocessor(
            self.image_config,
            fourier_position_encoding_kwargs=dict(
                max_resolution=(224, 224),
                num_bands=64
            )
        )
        self.image_model = PerceiverModel(self.image_config, input_preprocessor=self.image_preprocessor)
        self.image_model = self.image_model.to(self.device)

        self.name = 'Perceiver'
        
        self.transform = transforms.Compose([
                transforms.Lambda(lambda img: self.image_processor.preprocess(img, return_tensors="pt")['pixel_values']),
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
                outputs = self.text_model(inputs=text.input_ids, attention_mask=text.attention_mask)

                text_features.append(outputs['last_hidden_state'][:,0,:])

                images = torch.squeeze(images)
                images = images.to(self.device)
                outputs = self.image_model(inputs=images)

                image_features.append(outputs['last_hidden_state'][:,0,:])

            text_features = torch.vstack(text_features)
            text_features = torch.nn.functional.normalize(text_features, p=2.0, dim=1)

            image_features = torch.vstack(image_features)
            image_features = torch.nn.functional.normalize(image_features, p=2.0, dim=1)

        return text_features.cpu().squeeze(), image_features.cpu().squeeze()