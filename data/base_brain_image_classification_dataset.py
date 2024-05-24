from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from collections import Counter
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, split,data, img_offset_num=35, transform=None,is_inference=False):
        self.class_labels = ['AD', 'CN']
        self.transform = transform
        self.img_list = []
        self.label_list = []
        self.split=split
        if not isinstance(data, list):
            self.image_path = os.path.join(data, self.split)
            file_list=[os.path.join(self.image_path, img_name) for img_name in os.listdir(self.image_path)]
            print(f'Preparing {split} classification dataset')
            j=0
            for img_path in tqdm(set(file_list)):
                if j>100:
                    break
                j+=1
                img = np.load(img_path)
                img = np.array(img, np.uint8)
                for i in range(img_offset_num, len(img) - img_offset_num):
                    label = img_path.split('_')[-1].replace('.npy', '')
                    if label not in self.class_labels:
                        continue
                    self.img_list.append(self.transform(Image.fromarray(img[i])))
                    self.label_list.append(self.class_labels.index(label))

            unique_labels = sorted(set(self.label_list))
            label_to_index = {label: index for index, label in enumerate(unique_labels)}
            self.class_counts = [Counter(self.label_list)[label] for label in unique_labels]
            self.label_list = [np.array(int(label_to_index[label])) for label in self.label_list]
        else:
            self.img_list=[self.transform(Image.fromarray(sample)) for sample in data]
            self.label_list=[-1]
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        return self.img_list[index], self.label_list[index]

