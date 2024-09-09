import torch
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from transformers import CLIPTextConfig, CLIPVisionConfig, CLIPConfig, CLIPProcessor, CLIPModel
from models.custom_perceiver import ContrastiveLoss

from datetime import datetime
import wandb


class CustomCLIP():
    def __init__(self, pre_trained: bool):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #  wraps CLIPImageProcessor and CLIPTokenizer into a single instance to both encode the text and prepare the images.
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.text_tokenizer = self.processor
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: self.processor(images=img, return_tensors='pt')['pixel_values'])
        ])
        
        if pre_trained:
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.name = 'PretrainedCLIP'

        else:
            vision_config = CLIPVisionConfig(
                hidden_size=128,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=256,
            )

            text_config = CLIPTextConfig(
                hidden_size=128,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=256
            )
            
            config = CLIPConfig(
                vision_config=vision_config.to_dict(),
                text_config=text_config.to_dict(),
                projection_dim=512,
            )
            
            self.model = CLIPModel(config).to(self.device)
            self.name = 'CLIP'
        
    def train(self, dataloader, criterion, optimizer, grad_scaler):
        self.model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            
            images, text = batch
            optimizer.zero_grad()

            with autocast():
                text['input_ids'] = torch.flatten(text['input_ids'], start_dim=0, end_dim=1)
                text['attention_mask'] = torch.flatten(text['attention_mask'], start_dim=0, end_dim=1)
                text = text.to(self.device)
                text_embeds = self.model.get_text_features(
                    input_ids=text['input_ids'],
                    attention_mask=text['attention_mask']
                )

                images = torch.squeeze(images)
                if len(images.size()) == 3:
                    images = images.unsqueeze(0)
                images = images.to(self.device)
                img_embeds = self.model.get_image_features(images)

                loss = criterion(text_embeds, img_embeds)
                epoch_loss += loss.item()

            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            
        epoch_loss = epoch_loss / len(dataloader)

        return epoch_loss      

    def evaluation(self, dataloader, criterion):
        self.model.eval()
        epoch_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                images, text = batch

                text['input_ids'] = torch.flatten(text['input_ids'], start_dim=0, end_dim=1)
                text['attention_mask'] = torch.flatten(text['attention_mask'], start_dim=0, end_dim=1)
                text = text.to(self.device)
                text_embeds = self.model.get_text_features(
                    input_ids=text['input_ids'],
                    attention_mask=text['attention_mask']
                )
                
                images = torch.squeeze(images)
                if len(images.size()) == 3:
                    images = images.unsqueeze(0)
                images = images.to(self.device)
                img_embeds = self.model.get_image_features(images)

                loss = criterion(text_embeds, img_embeds)
                epoch_loss += loss.item()

            epoch_loss = epoch_loss / len(dataloader)

        return epoch_loss

    def orchestrate_training(self, dataset_name, train_dataloader, val_dataloader, test_dataloader,
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

        grad_scaler = GradScaler()

        wandb_config = {
            'number_of_parameters': sum(p.numel() for p in self.model.parameters()),
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
            name=f'Training CLIP {dataset_name} {current_date}'
            )

        for epoch in range(no_epochs):
            train_loss = self.train(train_dataloader, criterion, optimizer, grad_scaler)
            val_loss = self.evaluation(val_dataloader, criterion)
            print(f'Epoch: {epoch+1} --> train loss = {train_loss}, validation loss = {val_loss}')
            wandb.log({
                'epoch': epoch+1,
                'train_loss': train_loss,
                'val_loss': val_loss
            })

            scheduler.step()

        test_loss = self.evaluation(test_dataloader, criterion)

        wandb.log({'test_loss': test_loss})
        wandb.finish()

        torch.save(self.model.state_dict(), f'{save_path}/clip_{dataset_name}.pth')
        print(f'Saved model in {save_path}')

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
                    temp = self.model.get_text_features(
                        input_ids=text['input_ids'][:,i,:],
                        attention_mask=text['attention_mask'][:,i,:]
                        )
                    text_embeds.append(temp)
        
                text_embeds = torch.vstack(text_embeds)
                text_features.append(text_embeds)

                images = torch.squeeze(images)
                if len(images.size()) == 3:
                    images = images.unsqueeze(0)
                images = images.to(self.device)
                img_embeds = self.model.get_image_features(images)

                image_features.append(img_embeds)


            text_features = torch.vstack(text_features)
            text_features = torch.nn.functional.normalize(text_features, p=2.0, dim=1)

            image_features = torch.vstack(image_features)
            image_features = torch.nn.functional.normalize(image_features, p=2.0, dim=1)

        return text_features.cpu().squeeze(), image_features.cpu().squeeze()
    
    def encode_image(self, images):
        return self.model.get_image_features(images)

    def encode_text(self, text):
        return self.model.get_text_features(
                        input_ids=text['input_ids'],
                        attention_mask=text['attention_mask']
                        )