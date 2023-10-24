from .unet_models import UNET
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix

class MRIBrainImageSegmentationModel():
    def __init__(self, ckpt, batch_size=64, val_batch_size=128,
                 learning_rate=1e-4, epochs=500, es_paiteince=5, num_workers=8, train_ratio=0.85,device='gpu',gpu_ids=[0]):
        self.train_set = None
        self.test_set = None
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.es_paiteince = es_paiteince
        self.num_workers = num_workers
        self.model_ckpt = ckpt
        self.train_ratio = train_ratio
        self.device = device
        self.gpu_ids = gpu_ids

    def train_function(self, data, model, optimizer, loss_fn, device):
        # print('Entering into train function')
        model.train()
        data = tqdm(data)
        for index, batch in enumerate(data):
            X, y = batch
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = loss_fn(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()

    def normalize_image(self, image):
        # Normalize the image to a range of [0, 1]
        return (image - np.min(image)) / (np.max(image) - np.min(image)+1e-6)

    def compute_IoU(self,cm):
        '''
        Adapted from:
            https://github.com/davidtvs/PyTorch-ENet/blob/master/metric/iou.py
            https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/metrics.py#L2716-L2844
        '''

        sum_over_row = cm.sum(axis=0)
        sum_over_col = cm.sum(axis=1)
        true_positives = np.diag(cm)

        # sum_over_row + sum_over_col = 2 * true_positives + false_positives + false_negatives.
        denominator = sum_over_row + sum_over_col - true_positives

        iou = true_positives / denominator

        return iou, np.nanmean(iou)

    def eval_function(self,data, model, n_classes, device,is_inference=False):
        model.eval()
        labels = np.arange(n_classes)
        cm = np.zeros((n_classes, n_classes))
        data = tqdm(data)
        pred_images= []
        with torch.no_grad():
            for index, batch in enumerate(data):
                X, y = batch
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                probs = torch.softmax(outputs, dim=1)
                masks_pred = torch.argmax(probs, dim=1)
                for j in range(len(masks_pred)):
                    pred = masks_pred[j].cpu().detach().numpy().flatten()

                    if is_inference:
                        pred_image = self.normalize_image(masks_pred.cpu().numpy()[j, :, :])
                        pred_images.append(pred_image)
                    else:
                        true = y[j].cpu().detach().numpy().flatten()
                        cm += confusion_matrix(true, pred, labels=labels)

        if is_inference:
            return {'preds':pred_images}
        else:
            class_iou, mean_iou = self.compute_IoU(cm)
            eval_results = {'class_iou': class_iou,
                             'mean_iou': mean_iou,
                                }
            return eval_results

    def train(self, load_model=False):
        if self.train_set is None:
            raise ValueError("Train set is None. Please ensure a valid train set is provided.")
        loss_fn = nn.CrossEntropyLoss()
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

        self.model=self.model.to(DEVICE)
        loss_fn=loss_fn.to(DEVICE)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        mean_iou_best = 0
        if load_model:
            checkpoint = torch.load(self.model_ckpt)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
            mean_iou_best=checkpoint['mean_iou']
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
        n_p=0
        for epoch in range(self.epochs):
            print('Entering into epoch: ', epoch)
            train_loss = self.train_function(train_loader, self.model, optimizer, loss_fn, DEVICE)
            eval_results = self.eval_function(val_loader, self.model, self.n_classes, DEVICE)
            print('train_loss:', train_loss)
            print('mean_iou:', eval_results['mean_iou'])

            if eval_results['mean_iou'] > mean_iou_best:
                mean_iou_best = eval_results['mean_iou']
                n_p = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'mean_iou': eval_results['mean_iou'],
                    'train_loss': train_loss
                }, self.model_ckpt)
                print("Epoch completed and model successfully saved!")
            else:
                n_p += 1
            if n_p > self.es_paiteince:
                print('Early Stopping')
                break
        print('Training Finished')
        print('best_mean_iou', mean_iou_best)

    def fine_tune(self,learning_rate=1e-4):
        self.learning_rate = learning_rate
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
        eval_results = self.eval_function(test_loader, self.model, self.n_classes, DEVICE)
        print('Testing Finished')
        return eval_results

    def predict(self,inference_sample_dir):
        inference_set = self.dataset.get_inference_set(inference_sample_dir)
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
        self.model.to(DEVICE)

        checkpoint = torch.load(self.model_ckpt)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Model successfully loaded!")
        inference_loader = DataLoader(
            inference_set, batch_size=self.val_batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers)
        inference_outputs = self.eval_function(inference_loader, self.model, self.n_classes, DEVICE, is_inference=True)
        return inference_outputs

    def input_dataset(self, dataset):
        self.dataset = dataset
        self.train_set = dataset.train_set
        self.test_set = dataset.test_set
        self.n_classes = dataset.n_classes
        self.model = UNET(in_channels=1, classes=self.n_classes)


