from .base_feature_classification_model import BaseFeatureClassificationModel
import torch.nn as nn
import torch.nn.functional as F

class ComplexNN(nn.Module):
    def __init__(self,n_classes, input_dim, dropout_rate=0.5):
        super(ComplexNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

class MLPClassificationModel(BaseFeatureClassificationModel):
    def __init__(self, ckpt, batch_size=8, val_batch_size=8,
                 learning_rate=1e-4, epochs=500, es_paiteince=5, num_workers=8, train_ratio=0.85,device='gpu',gpu_ids=[0]):
        super().__init__(ckpt, batch_size, val_batch_size,
                 learning_rate, epochs, es_paiteince, num_workers, train_ratio,device,gpu_ids)

    def input_dataset(self, dataset):
        model = ComplexNN(n_classes=dataset.n_classes, input_dim=dataset.num_features)
        super().input_dataset(dataset, model)