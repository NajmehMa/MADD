from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, split='train', data_paths=['data_path1','data_path2'], transform=None,is_inference=False):
        self.transform = transform
        self.split = split
        self.yLabel_list = []
        self.eval = eval
        self.is_inference=is_inference
        self.is_list = True if not isinstance(data_paths[0], str) else False

        if not self.is_inference:
           self.mask_paths = [os.path.join(data_path, self.split, 'masks') for data_path in data_paths]
           self.mask_list = [os.path.join(mask_path, mask_name) for mask_path in self.mask_paths for mask_name in os.listdir(mask_path)]
        if not self.is_list:
            self.img_paths = [os.path.join(data_path, self.split, 'images') for data_path in data_paths]
            self.img_list = [os.path.join(img_path, img_name) for img_path in self.img_paths for img_name in os.listdir(img_path)]
        else:
            self.img_list =[sample for samples in data_paths for sample in samples]

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

        if not self.is_inference:
            mask = Image.open(self.mask_list[index])
            if self.transform is not None:
                image = self.transform(image)
                mask = self.transform(mask)
            return image, mask
        else:
            if self.transform is not None:
                image = self.transform(image)
            return image, image