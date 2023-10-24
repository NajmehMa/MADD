from .base_feature_classification_dataset import BaseFeatureDataset

class AgeDataset(BaseFeatureDataset):
    def __init__(self, data):
        super().__init__(data)

    def __repr__(self):
        return f"AgeDataset(train={len(self.train_set)}, test={len(self.test_set)})"





