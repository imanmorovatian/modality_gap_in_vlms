import wandb

import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from utils.clip import clip

from utils.custom_schedulers import get_cosine_schedule_with_warmup
from utils.contrastive_loss import compute_contrastive_loss


class CustomCLIP():
    def __init__(self, vision_encoder: str, pre_trained: bool):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if vision_encoder == 'RN50':
            self.model, self.transform = clip.load(name='RN50', pretrained=pre_trained, device=self.device, fp32=True)
            self.name = 'CLIP_RN50'
        else:
            self.model, self.transform = clip.load(name='ViT-B/32', pretrained=pre_trained, device=self.device, fp32=True)
            self.name = 'CLIP_ViT32'

        if pre_trained:
            self.name += '_Pre-trained'

        # if you want to reduce the number of layers of text transformer
        # self.model.transformer.layers = 6
        # self.model.transformer.resblocks = self.model.transformer.resblocks[:6]

        self._tokenizer = clip.tokenize

    def text_tokenizer(self, captions, *args, **kwargs):
        return self._tokenizer(texts=captions, context_length=77, truncate=True)

    def train(self, dataloader, optimizer, scheduler, grad_scaler):
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
                
                loss = compute_contrastive_loss(img_embeds, text_embeds, temperature)

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
                    
                    loss = compute_contrastive_loss(img_embeds, text_embeds, temperature)
                
                epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(dataloader)

        return epoch_loss

    def orchestrate_training(self, dataset_name, train_dataloader, val_dataloader, test_dataloader,
                             batch_size, no_epochs, save_path):
                
        optimizer = AdamW(self.model.parameters(), lr=5e-4, eps=1.0e-08, weight_decay=0.1)
        
        grad_scaler = GradScaler()

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

        torch.save(self.model.state_dict(), f'{save_path}/clip_{dataset_name}.pth')
        print(f'Saved model in {save_path}')

    def encode_for_retrieval(self, dataloader):
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
                
                loss = compute_contrastive_loss(img_embeds, text_embeds[::captions_per_image], temperature)
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