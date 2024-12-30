import torch
from torchvision import transforms
from transformers import AutoTokenizer, AutoProcessor, AlignModel


class CustomALIGN():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.text_tokenizer = AutoTokenizer.from_pretrained('kakaobrain/align-base')

        self.model = AlignModel.from_pretrained('kakaobrain/align-base')
        self.model = self.model.to(self.device)

        self.name = 'ALIGN'

        self._processor = AutoProcessor.from_pretrained('kakaobrain/align-base')
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: self._processor(images=img, return_tensors='pt')),
        ])
	
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
                
                batch_size, captions_per_image, _ = text['input_ids'].size()
                for ـ in range(batch_size):
                    # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
                    text_indices = list(range(text_index, text_index + captions_per_image))
                    image_to_text_map.append(text_indices)
                    text_index += captions_per_image

                    # Each of the next captions_per_image text captions correspond to the same image
                    text_to_image_map += [image_index] * captions_per_image
                    image_index += 1

                text['input_ids'] = torch.flatten(text['input_ids'], start_dim=0, end_dim=1)
                text['attention_mask'] = torch.flatten(text['attention_mask'], start_dim=0, end_dim=1)
                text = text.to(self.device)
                
                text_embeds = self.model.get_text_features(
                    input_ids = text['input_ids'],
                    attention_mask = text['attention_mask']).float()
                                
                images['pixel_values'] = torch.squeeze(images['pixel_values'])
                if len(images['pixel_values'].size()) == 3:
                    images['pixel_values'] = images['pixel_values'].unsqueeze(0)
                images = images.to(self.device)

                img_embeds = self.model.get_image_features(pixel_values=images['pixel_values']).float()


                image_features.append(img_embeds)
                text_features.append(text_embeds)
                
                temperature = self.model.temperature
                
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
