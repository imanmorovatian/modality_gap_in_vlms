import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import RobertaTokenizerFast, BridgeTowerImageProcessor, BridgeTowerModel


class CustomBridgeTower():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.text_tokenizer = RobertaTokenizerFast.from_pretrained("FacebookAI/roberta-base")
        self.image_preprocessor = BridgeTowerImageProcessor.from_pretrained("BridgeTower/bridgetower-base")
        self.model = BridgeTowerModel.from_pretrained('BridgeTower/bridgetower-base')
        self.model = self.model.to(self.device)
        self.transform = transforms.Compose([
                transforms.Lambda(lambda img: torch.tensor(self.image_preprocessor.preprocess(img)['pixel_values']) ),
                transforms.Resize((288, 288))
            ])
        
        self.name = 'BridgeTower'
    
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

                images = torch.squeeze(images)
                if len(images.size()) == 3:
                    images = images.unsqueeze(0)
                images = images.to(self.device)

                for i in range(no_captions):
                    input_temp = {key: value[:,i,:] for key, value in text.items()}
                    temp = self.model(input_ids=input_temp['input_ids'],
                                      attention_mask=input_temp['attention_mask'],
                                      pixel_values=images)
                    text_embeds.append(temp['text_features'][:,0,:])
                
                text_embeds = torch.vstack(text_embeds)
                text_features.append(text_embeds)

                img_embeds = self.model(input_ids=text['input_ids'][:,0,:],
                                      attention_mask=text['attention_mask'][:,0,:],
                                      pixel_values=images)['image_features'][:,0,:]
                image_features.append(img_embeds)

            text_features = torch.vstack(text_features)
            text_features = torch.nn.functional.normalize(text_features, p=2.0, dim=1)

            image_features = torch.vstack(image_features)
            image_features = torch.nn.functional.normalize(image_features, p=2.0, dim=1)

        return text_features.cpu().squeeze(), image_features.cpu().squeeze()