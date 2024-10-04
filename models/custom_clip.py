from PIL import Image
from functools import partial
import wandb

import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast
from torchvision import transforms
from transformers import CLIPModel

from models.clip_training import CLIP

from utils.simple_tokenizer import SimpleTokenizer
from utils.custom_schedulers import get_cosine_schedule_with_warmup
from utils.contrastive_loss import compute_contrastive_loss


def tokenize(captions, tokenizer, context_length=77, *args, **kwargs):
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    
    result = []
    for text in captions:
        tokens = [sot_token] + tokenizer.encode(text) + [eot_token]
        tokens = torch.Tensor(tokens)
        fixed_size_tokens = torch.zeros(context_length, dtype=torch.long)

        if len(tokens) >= context_length:
            fixed_size_tokens = tokens[:context_length]
        else:
            fixed_size_tokens[:len(tokens)] = tokens

        result.append(fixed_size_tokens)

    return torch.vstack(result)


class CustomCLIP():
    def __init__(self, pre_trained: bool, input_resolution=224):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        _tokenizer = SimpleTokenizer()
        self.text_tokenizer = partial(tokenize, tokenizer=_tokenizer)

        self.transform = transforms.Compose([
            transforms.Resize(input_resolution, interpolation=Image.BICUBIC),
            transforms.CenterCrop(input_resolution),
            lambda image: image.convert("RGB"),
            transforms.ToTensor(),
            transforms. Normalize((0.4225, 0.4012, 0.3659), (0.2681, 0.2635, 0.2763)),
        ])
        
        if pre_trained:
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.name = 'PretrainedCLIP'

        else:
            model_params = {
                'embed_dim': 1024,
                'image_resolution': 224,
                'vision_layers': (3, 4, 6, 3),
                'vision_width': 64,
                'vision_patch_size': None,
                'context_length': 77,
                'vocab_size': 49408,
                'transformer_width': 512,
                'transformer_heads': 8,
                'transformer_layers': 6
            }
            
            self.model = CLIP(**model_params).to(self.device)
            self.name = 'CLIP'
        
    def train(self, dataloader, optimizer, scheduler):
        self.model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            images, text = batch
            optimizer.zero_grad()

            with autocast():
                text = text[:,-1,:]
                text = text.to(self.device)

                images = torch.squeeze(images)
                if len(images.size()) == 3:
                    images = images.unsqueeze(0)
                images = images.to(self.device)

                img_embeds, text_embeds = self.model(images, text)                
                temperature = self.model.logit_scale.exp()
                
                loss = compute_contrastive_loss(img_embeds, text_embeds, temperature)
                loss.backward()
                epoch_loss += loss.item()
                optimizer.step()
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

                text = text[:,-1,:]
                text = text.to(self.device)

                images = torch.squeeze(images)
                if len(images.size()) == 3:
                    images = images.unsqueeze(0)
                images = images.to(self.device)

                img_embeds, text_embeds = self.model(images, text)                
                temperature = self.model.logit_scale.exp()
                
                loss = compute_contrastive_loss(img_embeds, text_embeds, temperature)
                epoch_loss += loss.item()

            epoch_loss = epoch_loss / len(dataloader)

        return epoch_loss

    def orchestrate_training(self, dataset_name, train_dataloader, val_dataloader, test_dataloader,
                             batch_size, no_epochs, save_path):
                
        optimizer = AdamW(self.model.parameters(), lr=5e-4, eps=1.0e-08, weight_decay=0.1)
        
        gradient_accumulation_steps = 1
        t_total = len(train_dataloader) // gradient_accumulation_steps * no_epochs
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
            name=f'Training CLIP {dataset_name}'
            )

        for epoch in range(no_epochs):
            train_loss = self.train(train_dataloader, optimizer, scheduler)
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

        torch.save(self.model.state_dict(), f'{save_path}/clip_{dataset_name}.pth')
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

                img_embeds, text_embeds = self.model(images, text)

                image_features.append(img_embeds)
                text_features.append(text_embeds)
                
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