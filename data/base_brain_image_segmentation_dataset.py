from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import torch

class SegmentationDataset(Dataset):
    def __init__(self, split, data, transform=None,is_inference=False):
        self.transform = transform
        self.split = split
        self.is_inference=is_inference
        self.is_list=True if isinstance(data, list) else False
        if not self.is_inference:
          self.mask_path = os.path.join(data, self.split, 'masks')
          self.mask_list = [os.path.join(self.mask_path, mask_name) for mask_name in os.listdir(self.mask_path)]
        if not self.is_list:
            self.img_path = self.label_path = os.path.join(data, self.split, 'images')
            self.img_list = [os.path.join(self.img_path, img_name) for img_name in os.listdir(self.img_path)]
        else:
            self.img_list=[sample for sample in data]

    def __len__(self):
        length = len(self.img_list)
        return length

    def __getitem__(self, index):
        if not self.is_list:
           image = Image.open(self.img_list[index])
        else:
            image=self.img_list[index]
            if isinstance(image, np.ndarray):
                image=Image.fromarray((image * 255).astype(np.uint8))

        if self.transform is not None:
            image = self.transform(image)
        if not self.is_inference:
            y = Image.open(self.mask_list[index])
            y = np.array(y)
            y = torch.from_numpy(y)
            y = y.type(torch.LongTensor)
            return image, y
        else:
            return image, image
