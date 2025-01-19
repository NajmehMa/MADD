from .base_skull_segmentation_dataset import SegmentationDataset
from torchvision import transforms
IMG_HEIGHT = 256
IMG_WIDTH = 256

class SkullImageSegmentationDataset:
    def __init__(self, data_paths=['data_path1','data_path2'],source='ADNI',type='Original-MRI-T1',version='0.0',distribution={'AD':0.3,'CN':0.7}):
        self.n_classes = 1
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.data_paths=data_paths
        self.train_set = SegmentationDataset(split='train', data_paths=self.data_paths, transform=self.transform)
        self.test_set = SegmentationDataset(split='test', data_paths=self.data_paths, transform=self.transform)
        self.info = {'data_points': {'train_len': len(self.train_set), 'test_len': len(self.test_set)},
                     'source': source,
                     'type': type,
                     'version': version,
                     'distribution': distribution,
                     }

    def get_train_dataset(self):
        return self.train_set

    def get_test_dataset(self):
        return self.test_set

    def get_inference_set(self, inference_sample):
        inference_set = SegmentationDataset(split='', data_paths=[inference_sample], transform=self.transform,is_inference=True)
        return inference_set

    def get_info(self):
        '''
        :return: A dictionary of the parameters info including: data_points, source, type, version, and distribution
        '''
        return self.info

    def set_info(self,info):
        '''
        :param info: A dictionary of the parameters info including: data_points, source, type, version, and distribution
        :return:
        '''
        self.info=info

    def add_dataset(self,data_paths):
        self.data_paths=self.data_paths+data_paths
        self.train_set = SegmentationDataset(split='train', data=self.data_paths, transform=self.transform)
        self.test_set = SegmentationDataset(split='test', data=self.data_paths, transform=self.transform)

    def __repr__(self):
        return f"SkullImageSegmentationDataset(train={len(self.train_set)}, test={len(self.test_set)})"
