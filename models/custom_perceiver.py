import numpy as np
from typing import Optional, Mapping, Tuple, Callable
import wandb

import torch
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms

from transformers import PerceiverConfig, PerceiverTokenizer, PerceiverImageProcessor, PerceiverModel
from transformers.models.perceiver.modeling_perceiver import PerceiverTextPreprocessor, PerceiverImagePreprocessor, PerceiverMultimodalPreprocessor

from utils.loss import compute_clip_loss
from utils.custom_schedulers import get_cosine_schedule_with_warmup


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
    
        
class CustomPerceiver():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.config = PerceiverConfig(
            num_latents=256, # default=256, reduced=256
            d_latents=768, # default=1280, reduced=512
            # d_model=128, # default=768, reduced=64
            num_self_attends_per_block=20, #default=26, reduced=4
            num_self_attention_heads=8, #default=8, reduced=4
            num_cross_attention_heads=8, #default=8, reduced=1
            image_size=224)
        
        self.preprocessor = PerceiverMultimodalPreprocessor(
            modalities={
                'text': PerceiverTextPreprocessor(self.config),
                'image': PerceiverImagePreprocessor(
                    self.config,
                    out_channels=128,
                    concat_or_add_pos='add',
                    fourier_position_encoding_kwargs=dict(
                    max_resolution=(224, 224),
                    num_bands=32,
                    concat_pos=False))
            },
            min_padding_size=0
        )
           
        self.model = PerceiverModel(self.config, input_preprocessor=self.preprocessor)
        self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.model = self.model.to(self.device)

        self.text_tokenizer = PerceiverTokenizer()
        self.image_transform = PerceiverImageProcessor()
        self.transform = transforms.Compose([
            transforms.Lambda(
                lambda img: self.image_transform.preprocess(
                    img,
                    input_data_format='channels_last',
                    return_tensors='pt')['pixel_values']
                )
        ])

        self.name = 'Perceiver'
        
    def train(self, dataloader, optimizer, scheduler, grad_scaler):
        self.model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            images, text = batch
            optimizer.zero_grad()

            with autocast():
                text['input_ids'] = text['input_ids'][:,-1,:]
                text['attention_mask'] = text['attention_mask'][:,-1,:]
                text = text.to(self.device)
                # text_embeds = self.model(
                #     inputs={'text': text['input_ids'],},
                #     attention_mask=text['attention_mask']
                # )['last_hidden_state'][:,0,:]

                images = torch.squeeze(images)
                if len(images.size()) == 3:
                    images = images.unsqueeze(0)
                images = images.to(self.device)
                # img_embeds = self.model(inputs={'image': images,})['last_hidden_state'][:,0,:]
                
                temp = self.model(inputs={'text': text['input_ids'], 'image': images},
                                  attention_mask=text['attention_mask'])
                
                temperature = self.model.logit_scale.exp()
                loss = compute_clip_loss(img_embeds, text_embeds, temperature)

            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            epoch_loss += loss.item()

            self.model.logit_scale.data = torch.clamp(self.model.logit_scale.data, 0, 4.6052)
            scheduler.step()
            
        epoch_loss = epoch_loss / len(dataloader)

        return epoch_loss

    def evaluation(self, dataloader):
        self.model.eval()
        epoch_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                images, text = batch

                with autocast():
                    text['input_ids'] = text['input_ids'][:,-1,:]
                    text['attention_mask'] = text['attention_mask'][:,-1,:]
                    text = text.to(self.device)
                    text_embeds = self.model(
                        inputs={'text': text['input_ids'],},
                        attention_mask=text['attention_mask']
                    )['last_hidden_state'][:,0,:]
                    
                    images = torch.squeeze(images)
                    if len(images.size()) == 3:
                        images = images.unsqueeze(0)
                    images = images.to(self.device)
                    img_embeds = self.model(inputs={'image': images,})['last_hidden_state'][:,0,:]

                    temperature = self.model.logit_scale.exp()
                    loss = compute_clip_loss(img_embeds, text_embeds, temperature)

                epoch_loss += loss.item()

            epoch_loss = epoch_loss / len(dataloader)

        return epoch_loss

    def orchestrate_training(self, dataset_name, train_dataloader, val_dataloader, test_dataloader,
                             batch_size, no_epochs, save_path):
        
        optimizer = optim.AdamW(self.model.parameters(), lr=5e-4, eps=1.0e-08, weight_decay=0.1)
        
        grad_scaler = GradScaler()

        gradient_accumulation_steps = 1
        t_total = len(train_dataloader) // gradient_accumulation_steps * no_epochs
        num_warmup_steps = int(0.20 * t_total)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
            )

        # wandb_config = {
        #     'number_of_parameters': sum(p.numel() for p in self.model.parameters()),
        #     'batch_size': batch_size,
        #     'number_of_epochs': no_epochs,
        # }

        # wandb.init(
        #     config=wandb_config,
        #     entity='iman_morovatian',
        #     project='Thesis',
        #     name=f'Training Perceiver {dataset_name}'
        #     )

        for epoch in range(no_epochs):
            train_loss = self.train(train_dataloader, optimizer, scheduler, grad_scaler)
            val_loss = self.evaluation(val_dataloader)
            print(f'Epoch: {epoch+1} --> train loss = {train_loss}, validation loss = {val_loss}')
            wandb.log({
                'epoch': epoch+1,
                'train_loss': train_loss,
                'val_loss': val_loss
            })


        test_loss = self.evaluation(test_dataloader)

        wandb.log({'test_loss': test_loss})
        wandb.finish()

        torch.save(self.model.state_dict(), f'{save_path}/perceiver_{dataset_name}.pth')
        print(f'Saved model in {save_path}')

    def encode_for_retrieval(self, dataloader, criterion):
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
            for batch in tqdm(dataloader):
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

                text = text.to(self.device)
                text_embeds = []
                for i in range(captions_per_image):
                    temp = self.model(
                        inputs={'text': text['input_ids'][:,i,:],},
                        attention_mask=text['attention_mask'][:,i,:]
                        )
                    text_embeds.append( temp['last_hidden_state'][:,0,:] )

                text_embeds = torch.vstack(text_embeds)
                text_features.append(text_embeds)

                images = torch.squeeze(images)
                if len(images.size()) == 3:
                    images = images.unsqueeze(0)
                images = images.to(self.device)
                img_embeds = self.model(inputs={'image': images,})['last_hidden_state'][:,0,:]

                image_features.append(img_embeds)

                loss = criterion(img_embeds, text_embeds[::captions_per_image])
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