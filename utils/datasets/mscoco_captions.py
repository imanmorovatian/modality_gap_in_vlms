import os
import random
import json
import pandas as pd
from PIL import Image
from typing import Callable, Optional

from torch.utils.data import Dataset


class MSCOCOCaptions(Dataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        no_cap_per_img = 1
    ) -> None:
        super(MSCOCOCaptions, self).__init__()
        self.name = 'MSCOCO'
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.annFile = os.path.expanduser(annFile)
        
        with open(self.annFile) as f:
            json_file = json.load(f)
            
            captions = json_file['annotations'] # image_id --> common, (id: id of caption), caption
            df_captions = pd.DataFrame(captions)

            images = json_file['images'] # license, file_name, coco_url, height, width, date_captured,
                                        # flickr_url, (id: id of image) --> common
            df_images = pd.DataFrame(images)
            df_images = df_images[['file_name', 'id']]
            
            df_img_cap = pd.merge(df_images, df_captions, left_on='id', right_on='image_id')
            df_img_cap = df_img_cap[['file_name', 'image_id', 'caption']]

            df_img_cap_final = pd.DataFrame(
                [
                    {
                        'file_name': info['file_name'].to_list()[0],
                        'caption': random.choices(info['caption'].to_list(), k=no_cap_per_img)
                    }
                    for img_id, info in df_img_cap.groupby('image_id')
                ]
            )

            self.annotations = df_img_cap_final.set_index('file_name')['caption'].to_dict()
            self.ids = list(self.annotations.keys())

    def __getitem__(self, index: int):
        """
        Args:
            index (int): index in [0, self.__len__())

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """

        img_id = self.ids[index]

        # Image
        filename = os.path.join(self.root, img_id)
        img = Image.open(filename).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Captions
        targets = self.annotations[img_id]
        
        # wanna limit the size of target here but error happened when search relevant
        # target = self.__remove_punctuation(target)
        # target = self.__limit_length(target)
        
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self) -> int:
        return len(self.ids)
    
    # def __remove_punctuation(self,texts):
    #     punctuation = string.punctuation
    #     translator = str.maketrans('', '', punctuation)
    #     for i, text in enumerate(texts):
    #         texts[i] = text.translate(translator)
    #     return texts
    
    # def __limit_length(self, captions, max_length=50):
    #     # limit the length of caption
    #     return [' '.join(caption.split()[:max_length]) for caption in captions]