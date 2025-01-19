from .unet_models import UNET
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
import numpy as np

class SkullImageSegmentationModel():
    def __init__(self,ckpt,batch_size=64,val_batch_size=128,
                 learning_rate=1e-4,epochs=500,es_paiteince=5,num_workers=8,train_ratio=0.85,device='gpu'
                 ,gpu_ids=[0,1],type='UNET',task='MRI-skull-stripping',version='0.0'):
        self.train_set=None
        self.test_set=None
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.es_paiteince = es_paiteince
        self.num_workers = num_workers
        self.model_ckpt=ckpt
        self.train_ratio=train_ratio
        self.device=device
        self.gpu_ids=gpu_ids
        self.config={'batch_size':self.batch_size,'val_batch_size':self.val_batch_size,
                     'learning_rate':self.learning_rate,'epochs':self.epochs,
                     'es_paiteince':self.es_paiteince,'num_workers':self.num_workers,
                     'model_ckpt':self.model_ckpt,'device':self.device,'gpu_ids':self.gpu_ids}

        self.info = {'type': type,
                'task': task,
                'version': version,
                }

    def get_info(self):
        '''
        :return: A dictionary of the parameters info including: type, task, and version
        '''
        return self.info

    def set_info(self,info):
        '''
        :param info: A dictionary of the parameters info including: type, task, and version
        '''
        self.info=info

    def get_config(self):
        '''
        :return: A dictionary of config parameters including:
        batch_size, val_batch_size, learning_rate, epochs, es_paiteince, num_workers,model_ckpt,train_ratio,device, and gpu_ids
        '''
        return self.config

    def set_config(self,config):
        '''
        :param config: A dictionary of config parameters including:
        batch_size, val_batch_size, learning_rate, epochs, es_paiteince, num_workers,model_ckpt,train_ratio,device, and gpu_ids
        '''
        self.batch_size = config['batch_size']
        self.val_batch_size = config['val_batch_size']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.es_paiteince = config['es_paiteince']
        self.num_workers = config['num_workers']
        self.model_ckpt = config['model_ckpt']
        self.train_ratio = config['train_ratio']
        self.device = config['device']
        self.gpu_ids = config['gpu_ids']
        self.config=config

    def train_function(self,data, model, optimizer, loss_fn, device):
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
    def normalize_image(self,image):
        # Normalize the image to a range of [0, 1]
        return (image - np.min(image)) / (np.max(image) - np.min(image)+1e-6)

    def eval_function(self,data, model, loss_fn, device,is_inference=False):
            model.eval()
            total_loss = 0
            count = 0
            pred_images=[]
            data = tqdm(data)
            with torch.no_grad():
                for index, batch in enumerate(data):
                    X, y = batch
                    X, y = X.to(device), y.to(device)
                    with torch.no_grad():
                        preds = model(X)
                    loss = loss_fn(preds, y)
                    total_loss += loss.item()
                    count += 1
                    if is_inference:
                        for j in range(len(preds)):
                            pred_image = self.normalize_image(preds.cpu().numpy()[j, 0, :, :])
                            pred_images.append(pred_image)
            if is_inference:
                return {'preds':pred_images}
            else:
                average_loss = total_loss / count
                eval_results={'loss':average_loss,}
                return eval_results



    def train(self,load_model=False):
        if self.train_set is None:
            raise ValueError("Train set is None. Please ensure a valid train set is provided.")
        loss_fn = nn.L1Loss()

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
        eval_loss_best = float('inf')
        if load_model:
            checkpoint = torch.load(self.model_ckpt)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
            eval_loss_best = checkpoint['eval_loss']
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
            eval_results = self.eval_function(val_loader, self.model, loss_fn, DEVICE)
            print('train_loss:', train_loss)
            print('eval_loss:', eval_results['loss'])

            if eval_results['loss'] < eval_loss_best:
                eval_loss_best = eval_results['loss']
                n_p = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'eval_loss': eval_results['loss'],
                    'train_loss': train_loss
                }, self.model_ckpt)
                print("Epoch completed and model successfully saved!")
            else:
                n_p += 1
            if n_p > self.es_paiteince:
                print('Early Stopping')
                break
        print('Training Finished')
        print('best_eval_loss', eval_loss_best)

    def fine_tune(self,learning_rate=1e-4):
        self.learning_rate = learning_rate
        self.train(load_model=True)

    def test(self):
        if self.test_set is None:
            raise ValueError("Test set is None. Please ensure a valid test set is provided.")
        loss_fn = nn.L1Loss()
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
        loss_fn.to(DEVICE)
        checkpoint = torch.load(self.model_ckpt)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Model successfully loaded!")
        test_loader = DataLoader(
            self.test_set, batch_size=self.val_batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers)
        eval_results = self.eval_function(test_loader, self.model, loss_fn, DEVICE)
        print('Testing Finished')
        return eval_results

    def predict(self,inference_sample_dir):
        inference_set = self.dataset.get_inference_set(inference_sample_dir)
        loss_fn = nn.L1Loss()
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
        loss_fn.to(DEVICE)
        checkpoint = torch.load(self.model_ckpt)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Model successfully loaded!")
        inference_loader = DataLoader(
            inference_set, batch_size=self.val_batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers)
        inference_outputs = self.eval_function(inference_loader, self.model, loss_fn, DEVICE,is_inference=True)
        return inference_outputs

    def input_dataset(self,dataset):
        self.dataset = dataset
        self.train_set=dataset.train_set
        self.test_set = dataset.test_set
        self.n_classes = dataset.n_classes
        self.model = UNET(in_channels=1, classes=self.n_classes)


