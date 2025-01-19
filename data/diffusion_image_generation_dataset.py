from .base_brain_image_classification_dataset import ImageDataset
from torchvision import transforms

class DiffusionImageGenerationDataset:
    def __init__(self, dataset_dir,image_size=(256,256)):
        self.n_classes = 2
        self.image_size=image_size
        train_transforms = transforms.Compose([
                               transforms.RandomApply([transforms.RandomOrder([
                                   transforms.RandomApply([transforms.ColorJitter(brightness=0.33, contrast=0.33, saturation=0.33, hue=0)],p=0.5),
                                   transforms.RandomApply([transforms.GaussianBlur((5, 5), sigma=(0.1, 1.0))],p=0.5),
                                   transforms.RandomApply([transforms.RandomHorizontalFlip(0.5)],p=0.5),
                                   transforms.RandomApply([transforms.RandomVerticalFlip(0.5)],p=0.5),
                                   transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.5),
                                ])],p=0.2),
                               transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean = 0.5,std = 0.5)
                           ])
        self.test_transforms = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))
            ]
        )
        self.train_set = ImageDataset(split='train', root_dir=dataset_dir, transform=train_transforms)
        self.test_set = ImageDataset(split='test', root_dir=dataset_dir, transform=self.test_transforms)

    def get_train_dataset(self):
        return self.train_set

    def get_test_dataset(self):
        return self.test_set

    def get_inference_set(self,inference_sample_dir):
        inference_set = ImageDataset(split='', root_dir=inference_sample_dir, transform=self.test_transforms)
        return inference_set

    def get_features(self):
        features={'train_len':len(self.train_set),
                  'test_len': len(self.test_set),
                  'class_labels': self.class_labels
        }
        return features

    def __repr__(self):
        return f"DiffusionImageGenerationDataset(train={len(self.train_set)}, test={len(self.test_set)})"






