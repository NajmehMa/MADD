from .base_feature_classification_model import BaseFeatureClassificationModel
import torch.nn as nn
import torch.nn.functional as F

class ComplexNN(nn.Module):
    def __init__(self,n_classes,input_dim):
        super(ComplexNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, n_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class PETNormalizedMeasClassificationModel(BaseFeatureClassificationModel):
    def __init__(self, ckpt, batch_size=64, val_batch_size=128,
                 learning_rate=1e-4, epochs=500, es_paiteince=5, num_workers=8, train_ratio=0.85,device='gpu',gpu_ids=[0]):
        super().__init__(ckpt, batch_size, val_batch_size,
                 learning_rate, epochs, es_paiteince, num_workers, train_ratio,device,gpu_ids)

    def input_dataset(self, dataset):
        model = ComplexNN(n_classes=dataset.n_classes, input_dim=dataset.num_features)
        super().input_dataset(dataset, model)


