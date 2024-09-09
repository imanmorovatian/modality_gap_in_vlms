import torch
from torch.utils.data import DataLoader

from models.custom_clip import CustomCLIP
from models.custom_perceiver import ContrastiveLoss

from utils.metrics.retrieval import CrossModalRetrieval
from utils.metrics.metrics import CMD, CD

from utils.datasets.flickr30k_captions import Flickr30kCaptions

metric_dict = {'cmr_uni': [],
                #'multimodal-space-xretrieval': [],
                'cmd': None,
                'cd': None,}
criterion = ContrastiveLoss(temperature=0.5)
model = CustomCLIP(pre_trained=True)
device = model.device
dataset = Flickr30kCaptions(root='data/images/flickr30k/',
                        annFile='data/annotations/flickr30k/test.token',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer,
                        no_cap_per_img=5)
loader = DataLoader(dataset, batch_size=128)

model.model.eval()
with torch.no_grad():
    # image_to_text_map[i] gives the corresponding text indices for the ith image
    #  (as there are multiple pieces of text for each image)
    image_to_text_map = []

    # text_to_image_map[i] gives the corresponding image index for the ith text
    text_to_image_map = []

    image_encodings = []
    text_encodings = []

    text_index = 0
    image_index = 0

    total_loss = 0
    total_batches = 0

    for images, text in loader:
        
        text = text.to(device)
        batch_size, captions_per_image, _ = text['input_ids'].size()

        images = torch.squeeze(images)
        if len(images.size()) == 3:
            images = images.unsqueeze(0)
        images = images.to(device)

        # Update text_to_image_map and image_to_text_map for this batch
        for i in range(batch_size):
            # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
            text_indices = list(range(text_index, text_index + captions_per_image))
            image_to_text_map.append(text_indices)
            text_index += captions_per_image

            # Each of the next captions_per_image text captions correspond to the same image
            text_to_image_map += [image_index] * captions_per_image
            image_index += 1

        for k in text.keys():
            text[k] = torch.flatten(text[k], start_dim=0, end_dim=1)
        
        image_encoding = model.encode_image(images)
        text_encoding = model.encode_text(text)

        image_encoding = image_encoding / image_encoding.norm(dim=-1, keepdim=True)
        text_encoding = text_encoding / text_encoding.norm(dim=-1, keepdim=True)

        # compute batch loss and update total loss (i.e. reduction='sum')
        loss = criterion(image_encoding, text_encoding[::5])
        total_loss += loss.item()
        total_batches += 1

        image_encodings.append(image_encoding)
        text_encodings.append(text_encoding)

    image_encodings = torch.cat(image_encodings)
    text_encodings = torch.cat(text_encodings)
    text_to_image_map = torch.LongTensor(text_to_image_map).to(device)
    image_to_text_map = torch.LongTensor(image_to_text_map).to(device)

    # Normalise encodings
    # image_encodings = image_encodings / image_encodings.norm(dim=-1, keepdim=True)
    # text_encodings = text_encodings / text_encodings.norm(dim=-1, keepdim=True)

    ######## METRICS ########
    for metric in metric_dict.keys():
        if metric == 'cmr_uni':
            retrieval_obj = CrossModalRetrieval(image_encodings=image_encodings,
                                                text_encodings=text_encodings,
                                                text_to_image_map=text_to_image_map,
                                                image_to_text_map=image_to_text_map,
                                                search_space='unimodal',
                                                k_vals=[1,5,10])
            metric_dict['cmr_uni'] = retrieval_obj.compute()

        elif metric == 'cmd':
            cmd = CMD()
            metric_dict['cmd'] = round(cmd(image_encodings, text_encodings[::5]).item(), 2)

        elif metric == 'cd':
            cd = CD()
            metric_dict['cd_img_txt'] = round(cd(image_encodings, text_encodings[::5]).item(), 2)


    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    metric_dict['avg_loss'] = avg_loss


for metric, value in metric_dict.items():
    print(f'{metric} --> {value}')