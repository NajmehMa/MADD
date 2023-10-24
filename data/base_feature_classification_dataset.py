import pandas as pd
import os
import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from collections import Counter

class BaseFeatureDataset:
    def __init__(self, data):

        self.train_df = pd.read_csv(os.path.join(data,'train.csv'))
        self.test_df = pd.read_csv(os.path.join(data,'test.csv'))
        self.class_labels=['AD', 'CN']
        self.n_classes = len(self.class_labels)
        # Filter data samples with labels 'AD' or 'CN'
        self.train_df = self.train_df[self.train_df['label'].isin(self.class_labels)]
        self.test_df = self.test_df[self.test_df['label'].isin(self.class_labels)]
        self.class_counts=[Counter(self.train_df['label'].values)[label] for label in self.class_labels]
        # Convert labels to integers
        self.train_df['label'] = self.train_df['label'].apply(lambda x: 0 if x == self.class_labels[0] else 1)
        self.test_df['label'] = self.test_df['label'].apply(lambda x: 0 if x == self.class_labels[0] else 1)

        self.train_df.replace(' ', 0, inplace=True)
        self.test_df.replace(' ', 0, inplace=True)
        self.train_df = self.train_df.applymap(lambda x: pd.to_numeric(x, errors='ignore'))
        self.test_df = self.test_df.applymap(lambda x: pd.to_numeric(x, errors='ignore'))

        feature_cols = self.train_df.columns.difference(['label'])
        # Create tensors
        scaler = StandardScaler()
        X_train = scaler.fit_transform(self.train_df[feature_cols])
        X_test = scaler.transform(self.test_df[feature_cols])
        # joblib.dump(scaler, scaler_to_save_dir)
        self.scaler=scaler
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(self.train_df['label'].values, dtype=torch.int64)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(self.test_df['label'].values, dtype=torch.int64)

        self.num_features = len(feature_cols)

        self.train_set = TensorDataset(X_train, y_train)
        self.test_set = TensorDataset(X_test, y_test)


    def get_train_dataset(self):
        return self.train_set

    def get_test_dataset(self):
        return self.test_set

    def get_inference_set(self,inference_sample):
        if not isinstance(inference_sample, list):
            inference_df = pd.read_csv(inference_sample)
            inference_df.replace(' ', 0, inplace=True)
            inference_df = inference_df.applymap(lambda x: pd.to_numeric(x, errors='ignore'))
            feature_cols = inference_df.columns.difference(['label'])
            inference_sample=inference_df[feature_cols]
        else:
            inference_sample=[self.class_labels.index(label) for label in inference_sample]
        X_inference = self.scaler.transform(inference_sample)
        X_inference = torch.tensor(X_inference, dtype=torch.float32)
        y_inference = torch.tensor(np.zeros(len(X_inference)), dtype=torch.int64)
        inference_set = TensorDataset(X_inference, y_inference)
        return inference_set

    def get_features(self):
        features={'train_len':len(self.train_set),
                  'test_len': len(self.test_set),
                  'class_labels': self.class_labels
        }
        return features




