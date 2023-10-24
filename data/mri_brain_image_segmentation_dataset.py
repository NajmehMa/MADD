from .base_brain_image_segmentation_dataset import SegmentationDataset
from torchvision import transforms
IMG_HEIGHT = 256
IMG_WIDTH = 256

class MRIBrainImageSegmentationDataset:
    def __init__(self, data,image_size=(256,256)):
        self.n_classes = 97
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])
        self.train_set = SegmentationDataset(split='train', data=data, transform=self.transform)
        self.test_set = SegmentationDataset(split='test', data=data, transform=self.transform)

    def get_train_dataset(self):
        return self.train_set

    def get_test_dataset(self):
        return self.test_set

    def get_inference_set(self, inference_sample):
        inference_set = SegmentationDataset(split='', data=inference_sample, transform=self.transform,is_inference=True)
        return inference_set

    def get_features(self):
        features={'train_len':len(self.train_set),
                  'test_len': len(self.test_set),
        }
        return features

    def __repr__(self):
        return f"MRIBrainImageSegmentationDataset(train={len(self.train_set)}, test={len(self.test_set)})"
