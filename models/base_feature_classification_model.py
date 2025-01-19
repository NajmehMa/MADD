import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
import torch.nn.functional as F


class BaseFeatureClassificationModel():
    def __init__(self, model_ckpt='', batch_size=64, val_batch_size=128,
                 learning_rate=1e-4, epochs=500, es_paiteince=5, num_workers=8, train_ratio=0.85,device='gpu',gpu_ids=[0]):
        self.train_set = None
        self.test_set = None
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.es_paiteince = es_paiteince
        self.num_workers = num_workers
        self.model_ckpt = model_ckpt
        self.train_ratio = train_ratio
        self.device=device
        self.gpu_ids=gpu_ids


    def weights_init(self,m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def train_function(self, data, model, optimizer, loss_fn, device):
        # print('Entering into train function')
        model.train()
        data = tqdm(data)
        for (features, labels) in data:
            features = features.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(features)
            outputs = F.softmax(outputs, dim=-1)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        return loss.item()



    def eval_function(self, data, model, device,is_inference=False):
        model.eval()
        scores = []
        valid_labels = []
        with torch.no_grad():
            for (features, labels) in data:
                features = features.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.long)
                outputs = model(features)
                outputs=F.softmax(outputs, dim=-1)
                scores.extend(outputs.cpu().numpy())
                valid_labels.extend(labels.cpu().numpy())
        if is_inference:
            return {'scores':[np.max(x) for x in scores],'preds':[self.class_labels[np.argmax(x)] for x in scores]}
        else:
            valid_outputs = [np.argmax(x) for x in scores]
            prec = precision_score(valid_labels, valid_outputs)
            rec = recall_score(valid_labels, valid_outputs)
            fscore = f1_score(valid_labels, valid_outputs)
            balanced_accuracy = balanced_accuracy_score(valid_labels, valid_outputs)

            eval_results = {'prec': prec,
                            'recall': rec,
                            'fscore': fscore,
                            'balanced_accuracy': balanced_accuracy,
                            }
            return eval_results

    def train(self, load_model=False):
        if self.train_set is None:
            raise ValueError("Train set is None. Please ensure a valid train set is provided.")

        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)

        if self.device == 'gpu' and torch.cuda.is_available():
            num_gpus = len(self.gpu_ids)
            DEVICE = torch.device('cuda')
            print(f'Running Train on {num_gpus} GPUs')
            if num_gpus > 1:
                print("Let's use", num_gpus, "GPUs!")
                self.model = nn.DataParallel(self.model, device_ids=self.gpu_ids)
        else:
            DEVICE = torch.device("cpu")
            print('Running Train on CPU')

        self.model.to(DEVICE)
        loss_fn.to(DEVICE)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        f_score_best = 0
        if load_model:
            checkpoint = torch.load(self.model_ckpt)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
            f_score_best = checkpoint['f_score']
            print("Model successfully loaded!")

        train_length = int(self.train_ratio * len(self.train_set))
        val_length = len(self.train_set) - train_length

        # Create new train and validation sets
        train_subset, val_subset = random_split(self.train_set, [train_length, val_length])

        # Now, create DataLoaders for train and validation subsets.
        train_loader = DataLoader(
            train_subset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers)
        val_loader = DataLoader(
            val_subset, batch_size=self.val_batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers)

        n_p = 0
        for epoch in range(self.epochs):
            print('Entering into epoch: ', epoch)
            train_loss = self.train_function(train_loader, self.model, optimizer, loss_fn, DEVICE)

            eval_results = self.eval_function(val_loader, self.model,DEVICE)
            print('train_loss:', train_loss)
            print('fscore:', eval_results['fscore'])
            print('balanced_accuracy:', eval_results['balanced_accuracy'])

            if eval_results['fscore'] > f_score_best:
                f_score_best = eval_results['fscore']
                n_p = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'f_score': eval_results['fscore'],
                    'train_loss': train_loss
                }, self.model_ckpt)
                print("Epoch completed and model successfully saved!")
            else:
                n_p += 1
            if n_p > self.es_paiteince:
                print('Early Stopping')
                break
        print('Training Finished')
        print('best_fscore',f_score_best)

    def fine_tune(self,learning_rate=1e-4):
        self.learning_rate=learning_rate
        self.train(load_model=True)

    def test(self):
        if self.test_set is None:
            raise ValueError("Test set is None. Please ensure a valid test set is provided.")
        if self.device == 'gpu' and torch.cuda.is_available():
            num_gpus = len(self.gpu_ids)
            DEVICE = torch.device('cuda')
            print(f'Running Test on {num_gpus} GPUs')
            if num_gpus > 1:
                print("Let's use", num_gpus, "GPUs!")
                self.model = nn.DataParallel(self.model, device_ids=self.gpu_ids)
        else:
            DEVICE = torch.device("cpu")
            print('Running Test on CPU')

        self.model.to(DEVICE)
        checkpoint = torch.load(self.model_ckpt)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Model successfully loaded!")
        test_loader = DataLoader(
            self.test_set, batch_size=self.val_batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers)
        test_results = self.eval_function(test_loader, self.model, DEVICE)
        print('Testing Finished')

        return test_results

    def predict(self,inference_sample_dir):
        inference_set=self.dataset.get_inference_set(inference_sample_dir)
        if self.device == 'gpu' and torch.cuda.is_available():
            num_gpus = len(self.gpu_ids)
            DEVICE = torch.device('cuda')
            print(f'Running Inference on {num_gpus} GPUs')
            if num_gpus > 1:
                print("Let's use", num_gpus, "GPUs!")
                self.model = nn.DataParallel(self.model, device_ids=self.gpu_ids)
        else:
            DEVICE = torch.device("cpu")
            print('Running Inference on CPU')
        checkpoint = torch.load(self.model_ckpt)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model=self.model.to(DEVICE)
        print("Model successfully loaded!")
        inference_loader = DataLoader(
            inference_set, batch_size=self.val_batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers)
        inference_outputs = self.eval_function(inference_loader, self.model, DEVICE,is_inference=True)
        return inference_outputs

    def input_dataset(self, dataset,model):
        self.dataset=dataset
        self.train_set = dataset.train_set
        self.test_set = dataset.test_set
        self.model = model
        self.model.apply(self.weights_init)
        n_samples = dataset.class_counts
        n_samples = np.array(n_samples)
        weights = 1.0 / n_samples
        weights = weights / np.sum(weights)
        w1, w2 = weights.tolist()
        self.class_weights = torch.tensor([w1, w2], dtype=torch.float)
        self.class_labels=dataset.class_labels



