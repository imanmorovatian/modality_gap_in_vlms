import os
import wandb

import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from utils.clip import clip

from utils.custom_schedulers import get_cosine_schedule_with_warmup


class CustomCLIP():
    def __init__(self,
                 vision_encoder: str,
                 frozen_text_encoder: bool = True,
                 text_encoder_from_local: bool = False,
                 frozen_image_encoder: bool = True,
                 image_encoder_from_local: bool = False,
                 frozen_projection_layers: bool = False,
                 projection_layers_from_local: bool = False,
                 local_pretrained_weights_path: str = None):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if vision_encoder == 'RN50':
            self.model, self.transform = clip.load(name='RN50', pretrained=True, device='cpu', fp32=True)
            self.name = 'CLIP_RN50'
        else:
            self.model, self.transform = clip.load(name='ViT-B/32', pretrained=True, device='cpu', fp32=True)
            self.name = 'CLIP_ViT32'
        
        self._tokenizer = clip.tokenize

        # if you want to reduce the number of layers of text transformer
        # self.model.transformer.layers = 6
        # self.model.transformer.resblocks = self.model.transformer.resblocks[:6]

        # By default, weights are loaded from the Internet. If you want to load some specific weights from
        # the local, you must specify using the *_from_local arguments
        if local_pretrained_weights_path is not None:
            local_state_dict = torch.load(local_pretrained_weights_path, map_location=torch.device('cpu'))
            hybrid_state_dict = {}

            if text_encoder_from_local:
                hybrid_state_dict['positional_embedding'] = local_state_dict['positional_embedding']
                hybrid_state_dict['ln_final.bias'] = local_state_dict['ln_final.bias']
                hybrid_state_dict['ln_final.weight'] = local_state_dict['ln_final.weight'] 
                hybrid_state_dict['token_embedding.weight'] = local_state_dict['token_embedding.weight']

                for name, param in local_state_dict.items():
                    if name.startswith('transformer'):
                        hybrid_state_dict[name] = param

            if image_encoder_from_local:
                for name, param in local_state_dict.items():
                    if name.startswith('visual'):
                        hybrid_state_dict[name] = param

            if projection_layers_from_local:
                hybrid_state_dict['text_projection'] = local_state_dict['text_projection']
                hybrid_state_dict['visual.proj'] = local_state_dict['visual.proj']

            filled_params = set(hybrid_state_dict.keys())
            for name, param in self.model.named_parameters():
                if name not in filled_params:
                    hybrid_state_dict[name] = param

            self.model.load_state_dict(hybrid_state_dict)

        
        if frozen_text_encoder:
            self.model.positional_embedding.requires_grad = False
            self.model.ln_final.bias.requires_grad = False
            self.model.ln_final.weight.requires_grad = False
            self.model.token_embedding.weight.requires_grad = False

            for name, param in self.model.named_parameters():
                if name.startswith('transformer'):
                    param.requires_grad = False

        if frozen_image_encoder:
            for name, param in self.model.named_parameters():
                if name.startswith('visual'):
                    param.requires_grad = False

        if frozen_projection_layers:
            self.model.text_projection.requires_grad = False
            self.model.visual.proj.requires_grad = False

        self.model = self.model.to(self.device)


        if frozen_image_encoder:
            self.name += '_L' # (L)ocked 
            if image_encoder_from_local:
                self.name += 'l' # loaded from (l)ocal
            else:
                self.name += 'i' # loaded from the (I)nternet
        else:
            self.name += '_U' # (U)nlocked
            if image_encoder_from_local:
                self.name += 'l' # loaded from (l)ocal
            else:
                self.name += 'i' # loaded from the (I)nternet

        if frozen_text_encoder:
            self.name += 'L' # (L)ocked 
            if text_encoder_from_local:
                self.name += 'l' # loaded from (l)ocal
            else:
                self.name += 'i' # loaded from the (I)nternet
        else:
            self.name += 'U' # (U)nlocked
            if text_encoder_from_local:
                self.name += 'l' # loaded from (l)ocal
            else:
                self.name += 'i' # loaded from the (I)nternet
            
    def text_tokenizer(self, captions, *args, **kwargs):
        return self._tokenizer(texts=captions, context_length=77, truncate=True)

    def train(self, dataloader, loss_function, optimizer, scheduler, grad_scaler):
        self.model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            images, text = batch
            optimizer.zero_grad()

            with autocast():
                # for the training, one caption per image is used
                text = text[:,-1,:]
                text = text.to(self.device)

                images = torch.squeeze(images)
                if len(images.size()) == 3:
                    images = images.unsqueeze(0)
                images = images.to(self.device)

                img_embeds = self.model.encode_image(images)
                text_embeds = self.model.encode_text(text)                
                temperature = self.model.logit_scale.exp()
                
                loss = loss_function(img_embeds, text_embeds, temperature)

            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            epoch_loss += loss.item()

            self.model.logit_scale.data = torch.clamp(self.model.logit_scale.data, 0, 4.6052)
            scheduler.step() 
            
        epoch_loss = epoch_loss / len(dataloader)

        return epoch_loss      

    def evaluation(self, dataloader, loss_function):
        self.model.eval()
        epoch_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                images, text = batch

                text = text[:,-1,:]
                text = text.to(self.device)

                images = torch.squeeze(images)
                if len(images.size()) == 3:
                    images = images.unsqueeze(0)
                images = images.to(self.device)

                with autocast():
                    img_embeds = self.model.encode_image(images)
                    text_embeds = self.model.encode_text(text)                
                    temperature = self.model.logit_scale.exp()
                    
                    loss = loss_function(img_embeds, text_embeds, temperature)
                
                epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(dataloader)

        return epoch_loss

    def orchestrate_training(self, dataset_name, train_dataloader, val_dataloader, test_dataloader,
                             loss_function, batch_size, no_epochs):
                
        optimizer = AdamW(self.model.parameters(), lr=5e-4, eps=1.0e-08, weight_decay=0.1)
        
        grad_scaler = GradScaler()

        t_total = len(train_dataloader) * no_epochs
        
        if self.name.split('_')[2][0]== 'L' and self.name.split('_')[2][0] == 'L':
            # self.name.split('_')[2] is (L or U)(l or I)(L or U)(l or I)
            # so self.name.split('_')[2][0]== 'L' and self.name.split('_')[2][0] == 'L' means
            # just finetunning the projection layers

            num_warmup_steps = 0
        else:
            num_warmup_steps = int(0.20 * t_total)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
            )
        
        wandb_config = {
            'number_of_parameters': sum(p.numel() for p in self.model.parameters()),
            'batch_size': batch_size,
            'number_of_epochs': no_epochs,
        }

        wandb.init(
            config=wandb_config,
            entity='iman_morovatian',
            project='Thesis',
            name=f'Training {self.name} {dataset_name} {loss_function.__name__}'
            )

        for epoch in range(no_epochs):
            train_loss = self.train(train_dataloader, loss_function, optimizer, scheduler, grad_scaler)
            val_loss = self.evaluation(val_dataloader, loss_function)
            
            print(f'Epoch: {epoch+1} --> train loss = {train_loss}, validation loss = {val_loss}')
            wandb.log({
                'epoch': epoch+1,
                'train_loss': train_loss,
                'val_loss': val_loss
            })

        test_loss = self.evaluation(test_dataloader, loss_function)

        wandb.log({'test_loss': test_loss})
        wandb.finish()

        model_name, vision_encoder, ext_name = self.name.split('_')
        loss_name = loss_function.__name__.split('compute_')[-1]
        folder = f'weights/{model_name}'
        if not os.path.exists(folder):
            os.makedirs(folder)

        torch.save(self.model.state_dict(), f'{folder}/{vision_encoder}_{ext_name}_{loss_name}_{dataset_name}.pth')
        print(f'Saved model in {folder}')

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
                
                images = torch.squeeze(images)
                if len(images.size()) == 3:
                    images = images.unsqueeze(0)
                images = images.to(self.device)

                img_embeds = self.model.encode_image(images)
                text_embeds = self.model.encode_text(text) 

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