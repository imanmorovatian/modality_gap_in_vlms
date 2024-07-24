import numpy as np
from typing import Optional, Mapping, Tuple, Callable
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import PerceiverConfig, PerceiverTokenizer, PerceiverImageProcessor, PerceiverModel
from transformers.models.perceiver.modeling_perceiver import PerceiverTextPreprocessor, PerceiverImagePreprocessor, PerceiverMultimodalPreprocessor, PerceiverModelOutput
from datetime import datetime
import wandb
from tqdm import tqdm


PreprocessorOutputType = Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]
PreprocessorType = Callable[..., PreprocessorOutputType]
class SharedPerceiverPreprocessor(PerceiverMultimodalPreprocessor):
    def __init__(self,
        modalities: Mapping[str, PreprocessorType],
        mask_probs: Optional[Mapping[str, float]] = None,
        min_padding_size: int = 2):
        super(SharedPerceiverPreprocessor, self).__init__(
            modalities, mask_probs, min_padding_size
        )
        
    def forward(
        self, inputs: Mapping[str, torch.Tensor], pos: Optional[torch.Tensor] = None,
        network_input_is_1d: bool = True, *args, **kwargs) -> PreprocessorOutputType:
        for modality, data in inputs.items():
            inputs, modality_sizes, inputs_without_pos = self.modalities[modality](data)

        return inputs, modality_sizes, inputs_without_pos
    
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
    
    def forward(self, img_embeds, txt_embeds):

        batch_size = img_embeds.size(0)
        labels = torch.arange(batch_size).to(img_embeds.device)
        
        logits = self.cosine_similarity(img_embeds.unsqueeze(1), txt_embeds.unsqueeze(0)) * np.exp(self.temperature)
        
        loss_img = F.cross_entropy(logits, labels)
        loss_txt = F.cross_entropy(logits.T, labels)
        
        return (loss_img + loss_txt) / 2
    
class CustomPerceiver():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.config = PerceiverConfig(d_model=64, d_latents=1024, image_size=224, qk_channels=1024)
        self.preprocessor = SharedPerceiverPreprocessor(
            modalities={
                'text': PerceiverTextPreprocessor(self.config),
                'image': PerceiverImagePreprocessor(self.config,
                                                    out_channels=64,
                                                    concat_or_add_pos='add',
                                                    fourier_position_encoding_kwargs=dict(
                                                    max_resolution=(224, 224),
                                                    num_bands=16,
                                                    concat_pos=False))
            },
            min_padding_size=0
        )   
        self.text_tokenizer = PerceiverTokenizer()
        self.model = PerceiverModel(self.config, input_preprocessor=self.preprocessor).to(self.device)

        self.image_transform = PerceiverImageProcessor()
        
        self.name = 'Perceiver'
        
        self.transform = transforms.Compose([
                transforms.Lambda(lambda img: self.image_transform.preprocess(img, return_tensors="pt")['pixel_values'])
            ])
    
    def train(self, dataset, batch_size, criterion, optimizer):
        self.model.train()
        running_loss = 0.0
        dataloader = DataLoader(dataset, batch_size=batch_size)
        for batch in tqdm(dataloader):
            images, text = batch

            text = text[0]
            text = self.text_tokenizer(text, padding=True, truncation=True, return_tensors='pt')
            text = text.to(self.device)
            text_embeds = self.model(inputs={'text': text.input_ids,})['last_hidden_state'][:,0,:]

            images = torch.squeeze(images)
            images = images.to(self.device)
            img_embeds = self.model(inputs={'image': images,})['last_hidden_state'][:,0,:]

            optimizer.zero_grad()
            loss = criterion(img_embeds, text_embeds)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)

        return epoch_loss

    def evaluation(self, dataset, batch_size, criterion, optimizer):
        self.model.eval()
        running_loss = 0.0
        dataloader = DataLoader(dataset, batch_size=batch_size)
        with torch.no_grad():
            for batch in tqdm(dataloader):
                images, text = batch

                text = text[0]
                text = self.text_tokenizer(text, padding=True, truncation=True, return_tensors='pt')
                text = text.to(self.device)
                text_embeds = self.model(inputs={'text': text.input_ids,})
                text_embeds = text_embeds['last_hidden_state'][:,0,:]

                images = torch.squeeze(images)
                images = images.to(self.device)
                img_embeds = self.model(inputs={'image': images,})['last_hidden_state'][:,0,:]

                loss = criterion(img_embeds, text_embeds)

                running_loss += loss.item()

            epoch_loss = running_loss / len(dataloader)

        return epoch_loss

    def orchestrate_training(self, dataset_name, train_dataset, val_dataset, test_dataset,
                             batch_size, no_epochs, save_path):
        
        criterion = ContrastiveLoss(temperature=0.5)

        opt_lr = 1e-3
        opt_wd = 1e-4
        optimizer = optimizer = optim.Adam(
            self.model.parameters(),
            lr=opt_lr,
            weight_decay=opt_wd
            )
        
        gamma_value = 0.95
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma_value)

        wandb_config = {
            'batch_size': batch_size,
            'number_of_epochs': no_epochs,
            'optimizer': 'Adam',
            'optimizer_learning_rate': opt_lr,
            'optimizer_weight_decay': opt_wd,
            'scheduler': 'ExponentialLR',
            'scheduler_gamma': gamma_value
        }

        current_date = datetime.today().strftime('%Y-%m-%d')
        wandb.init(
            config=wandb_config,
            entity='iman_morovatian',
            project='Thesis',
            name=f'Training Perceiver {dataset_name} {current_date}'
            )

        for epoch in range(no_epochs):
            train_loss = self.train(train_dataset, batch_size, criterion, optimizer)
            val_loss = self.evaluation(val_dataset, batch_size, criterion, optimizer)

            wandb.log({
                'epoch': epoch+1,
                'train_loss': train_loss,
                'val_loss': val_loss
            })

            scheduler.step()

        test_loss = self.evaluation(test_dataset, batch_size, criterion, optimizer)

        wandb.log({'test_loss': test_loss})
        wandb.finish()

        torch.save(self.model.state_dict(), f'{save_path}/perceiver.pth')
        
    # def encode(self, dataset, batch_size):
    #     dataloader = DataLoader(dataset, batch_size=batch_size)

    #     image_features = []
    #     text_features = []

    #     with torch.no_grad():
    #         for batch in tqdm(dataloader):
    #             images, text = batch

    #             text = text[0]
    #             text = self.text_tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    #             text = text.to(self.device)
    #             outputs = self.text_model(inputs=text.input_ids, attention_mask=text.attention_mask)

    #             text_features.append(outputs['last_hidden_state'][:,0,:])

    #             images = torch.squeeze(images)
    #             images = images.to(self.device)
    #             outputs = self.image_model(inputs=images)

    #             image_features.append(outputs['last_hidden_state'][:,0,:])

    #         text_features = torch.vstack(text_features)
    #         text_features = torch.nn.functional.normalize(text_features, p=2.0, dim=1)

    #         image_features = torch.vstack(image_features)
    #         image_features = torch.nn.functional.normalize(image_features, p=2.0, dim=1)

    #     return text_features.cpu().squeeze(), image_features.cpu().squeeze()