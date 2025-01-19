from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from collections import Counter
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, split='train',data_paths=['data_path1','data_path2'], img_offset_num=35, transform=None,is_inference=False):
        self.class_labels = ['AD', 'CN']
        self.transform = transform
        self.img_list = []
        self.label_list = []
        self.split=split
        if isinstance(data_paths[0], str):
            self.img_paths = [os.path.join(data_path, self.split) for data_path in data_paths]
            file_list = [os.path.join(img_path, img_name) for img_path in self.img_paths for img_name in os.listdir(img_path)]
            print(f'Preparing {split} classification dataset')
            for img_path in tqdm(set(file_list)):
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
            self.img_list=[self.transform(Image.fromarray(sample)) for samples in data_paths for sample in samples]
            self.label_list=[-1 for samples in data_paths for _ in samples]
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        return self.img_list[index], self.label_list[index]

