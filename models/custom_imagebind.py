import torch
from torchvision import transforms

from utils.ImageBind import data
from utils.ImageBind.models import imagebind_model
from utils.ImageBind.models.imagebind_model import ModalityType

from tqdm import tqdm


class CustomImageBind():
    def __init__(self, pretrained=True):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = imagebind_model.imagebind_huge(pretrained=pretrained)
        self.model.to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        self.name = 'ImageBind'
    
    def text_tokenizer(self, text, *args, **kwargs):
        return {ModalityType.TEXT: data.load_and_transform_text(text, self.device)}
    
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
            for batch in tqdm(dataloader, total=len(dataloader)):
                images, text = batch
                text = text['text']
                
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

                img_embeds = self.model({ModalityType.VISION: images})[ModalityType.VISION]
                text_embeds = self.model({ModalityType.TEXT: text})[ModalityType.TEXT]

                image_features.append(img_embeds)
                text_features.append(text_embeds)
                
                temperature = self.model.modality_postprocessors.text[1].log_logit_scale.exp()
                
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