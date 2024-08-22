import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel


class CustomCLIP():
    def __init__(self, pre_trained: bool):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if pre_trained:
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)

            #  wraps CLIPImageProcessor and CLIPTokenizer into a single instance to both encode the text and prepare the images.
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.text_tokenizer = self.processor
            self.transform = transforms.Compose([
                transforms.Lambda(lambda img: self.processor(images=img, return_tensors='pt')['pixel_values'])
            ])

            self.name = 'PretrainedCLIP'
        else:
            self.name = 'CLIP'
        
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