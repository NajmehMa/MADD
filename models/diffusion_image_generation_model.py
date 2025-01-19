import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
import numpy as np
from scipy.linalg import sqrtm
from sklearn.metrics import precision_score,recall_score,f1_score,balanced_accuracy_score
from .resnet_models import resnet18
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import inception_v3

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return x + self.block(x)

class DiffusionModel(nn.Module):
    def __init__(self, num_timesteps,num_classes):
        self.num_classes = num_classes
        super(DiffusionModel, self).__init__()
        self.num_timesteps = num_timesteps

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.residual_blocks = nn.Sequential(
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def one_hot(self,labels, num_classes):
        return torch.eye(num_classes)[labels].to(labels.device)
    def forward(self, x, labels, beta_t):
        # One-hot encode labels
        labels_one_hot = self.one_hot(labels, self.num_classes)
        b, c, h, w = x.size()
        labels_expanded = labels_one_hot[:, :, None, None].expand(-1, -1, h, w)

        for t in range(self.num_timesteps):
            noise = torch.randn_like(x) * torch.sqrt(beta_t[t].clone().detach())
            x = (torch.sqrt(1 - beta_t[t]) * x) + noise

            # Incorporate class labels into x (simple concatenation as an example)
            x_with_labels = torch.cat([x, labels_expanded], dim=1)
            x = self.encoder(x_with_labels)
            x = self.residual_blocks(x)
            x = self.decoder(x)
        return x

class BrainImageClassificationModel():
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
        self.device=device
        self.gpu_ids=gpu_ids

    def train_function(self, data, model,classifier_model, optimizer, loss_fn,class_criterion, beta_t,device):
        # print('Entering into train function')
        model.train()
        classifier_model.eval()
        data = tqdm(data)
        total_loss=0
        for index, (img, label) in enumerate(data):
            real_images = img.to(device=device, dtype=torch.float32)
            labels = label.to(device=device, dtype=torch.long)

            # Reset gradients
            optimizer.zero_grad()

            # Generate noisy images using diffusion model
            noisy_images = model(real_images, labels, beta_t)

            # Compute loss (MSE between real images and their noisy version)
            id_loss = loss_fn(noisy_images, real_images)

            with torch.no_grad():  # no need to compute gradients for classifier
                classifier_output = classifier_model(noisy_images)
            logits = F.log_softmax(classifier_output, dim=-1)

            class_loss = class_criterion(logits, labels.to(dtype=torch.long))

            loss = id_loss + 0.5 * class_loss
            total_loss += loss.item()
            # Backward pass
            loss.backward()
            optimizer.step()

        return total_loss
    def normalize_image(self,image):
        # Normalize the image to a range of [0, 1]
        return (image - np.min(image)) / (np.max(image) - np.min(image)+1e-6)

    def reshape_and_repeat(self,images_list, image_size):
        # Reshape and repeat grayscale images for FID calculation
        num_images = len(images_list)
        images = np.array(images_list).reshape(num_images, 1, image_size[0], image_size[1])
        return np.repeat(images, 3, axis=1)  # Repeat grayscale channel 3 times

    def torch_cov(self,m, rowvar=False):
        '''Estimate a covariance matrix given data.

        Covariance indicates the level to which two variables vary together.
        If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
        then the covariance matrix element `C_{ij}` is the covariance of
        `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

        Args:
            m: A 1-D or 2-D array containing multiple variables and observations.
                Each row of `m` represents a variable, and each column a single
                observation of all those variables.
            rowvar: If `rowvar` is True, then each row represents a
                variable, with observations in the columns. Otherwise, the
                relationship is transposed: each column represents a
                variable, while the rows contain observations.

        Returns:
            The covariance matrix of the variables.
        '''
        if m.dim() > 2:
            raise ValueError('m has more than 2 dimensions')
        if m.dim() < 2:
            m = m.view(1, -1)
        if not rowvar and m.size(0) != 1:
            m = m.t()
        # m = m.type(torch.double)  # uncomment this line if desired
        fact = 1.0 / (m.size(1) - 1)
        m -= torch.mean(m, dim=1, keepdim=True)
        mt = m.t()  # if complex: mt = m.t().conj()
        return fact * m.matmul(mt).squeeze()

    def calculate_fid(self,images1, images2, device):
        # Convert to PyTorch tensors
        images1 = torch.tensor(images1).float().to(device)
        images2 = torch.tensor(images2).float().to(device)

        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        preprocessed_images1 = []
        preprocessed_images2 = []
        for i in range(images1.size(0)):  # assuming images1 and images2 have the same batch size
            img1 = preprocess(images1[i].cpu()).to(device)
            img2 = preprocess(images2[i].cpu()).to(device)
            preprocessed_images1.append(img1)
            preprocessed_images2.append(img2)

        images1 = torch.stack(preprocessed_images1)
        images2 = torch.stack(preprocessed_images2)
        # Load inception model
        inception_model = inception_v3(pretrained=True, transform_input=False).eval().to(device)  # Move model to device

        # Extract features for both image sets
        with torch.no_grad():
            pred1 = inception_model(images1)
            pred2 = inception_model(images2)

        # Compute mean and covariance for both sets
        mu1, sigma1 = pred1.mean(0), self.torch_cov(pred1)
        mu2, sigma2 = pred2.mean(0), self.torch_cov(pred2)

        # Compute sum of squared difference between the means
        ssdiff = torch.sum((mu1 - mu2) ** 2.0)

        # Compute sqrt of product between cov
        covmean = sqrtm((sigma1.cpu().numpy() + sigma2.cpu().numpy()) / 2)

        # Check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        # Compute the final FID score
        fid = ssdiff + torch.trace(sigma1 + sigma2 - 2 * torch.tensor(covmean).to(sigma1.device))

        return fid.item()

    def eval_function(self,data, model, classifier_model, beta_t,device,is_inference=False):
        model.eval()
        classifier_model.eval()
        real_images_dict = {0: [], 1: []}
        fake_images_dict = {0: [], 1: []}
        gen_imgs_dict={}
        balanced_accuracies=[]
        # Loop through test_loader to generate fake images with corresponding noise and labels
        for i, (real_images, labels) in enumerate(data):

            real_images, labels = real_images.to(device), labels.to(device)
            batch_size = real_images.size(0)

            # Generate fake images using diffusion model
            fake_images = model(real_images, labels, beta_t)
            # Forward pass the noisy images through the classifier model
            with torch.no_grad():  # no need to compute gradients for classifier
                classifier_output = classifier_model(fake_images)
            logits = F.log_softmax(classifier_output, dim=-1)
            # Compute balanced classification accuracy
            _, pred_labels = torch.max(logits, dim=-1)
            # print(classifier_output)
            # print(pred_labels)
            true_labels = labels.to(dtype=torch.long).cpu().numpy()
            # print(pred_labels.cpu().numpy(),true_labels)
            balanced_acc = balanced_accuracy_score(true_labels, pred_labels.cpu().numpy())
            balanced_accuracies.append(balanced_acc)

            for j in range(batch_size):
                label = labels[j].item()
                real_images_dict[label].append(real_images[j])
                fake_images_dict[label].append(fake_images[j])

        # Compute FID scores separately for each class
        fid_scores = {}
        for label in [0, 1]:
            real_images = real_images_dict[label]
            fake_images = fake_images_dict[label]
            real_imgs = []
            gen_imgs = []
            for real_img, fake_img in zip(real_images, fake_images):
                gen_image = self.normalize_image(fake_img.cpu().numpy()[0, :, :])
                real_img = self.normalize_image(real_img.cpu().numpy()[0, :, :])

                real_imgs.append(real_img)
                gen_imgs.append(gen_image)


            # Reshape and repeat images for FID calculation
            real_imgs = self.reshape_and_repeat(real_imgs, self.image_size)
            gen_imgs = self.reshape_and_repeat(gen_imgs, self.image_size)

            # Calculate and print FID score
            fid_score = self.calculate_fid(real_imgs, gen_imgs, device)
            fid_scores[label] = fid_score
            gen_imgs_dict[label]=gen_imgs
        # Compute average balanced classification accuracy for the epoch
        avg_balanced_acc = sum((balanced_accuracies) / len(balanced_accuracies))

        if is_inference:
            return {'preds':gen_imgs_dict}
        else:

            eval_results = {
                            'fid_scores': fid_scores,
                            'balanced_accuracy': avg_balanced_acc,
                            }
            return eval_results

    def train(self, load_model=False):
        if self.train_set is None:
            raise ValueError("Train set is None. Please ensure a valid train set is provided.")

        loss_fn = nn.MSELoss() # Using MSE as an example
        class_criterion = nn.CrossEntropyLoss(weight=self.class_weights)

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
        self.classifier_model=self.classifier_model.to(DEVICE)
        loss_fn=loss_fn.to(DEVICE)
        class_criterion=class_criterion.to(DEVICE)
        beta_t = [0.001 * (i + 1) for i in range(self.num_timesteps)]
        beta_t = torch.tensor(beta_t).to(DEVICE)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_fid=float('inf')
        if load_model:
            checkpoint = torch.load(self.model_ckpt)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
            best_fid = checkpoint['best_fid']
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

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
        n_p=0
        for epoch in range(self.epochs):
            print('Entering into epoch: ', epoch)
            train_loss = self.train_function(train_loader, self.model,self.classifier_model, optimizer, loss_fn,class_criterion, beta_t,DEVICE)

            eval_results = self.eval_function(val_loader, self.model,self.classifier_model, beta_t,DEVICE)
            print('train_loss:', train_loss)
            print('fid_score_class_0:', eval_results['fid_scores'][0])
            print('fid_score_class_1:', eval_results['fid_scores'][1])
            print('balanced_accuracy:', eval_results['balanced_accuracy'])

            if sum(eval_results['fid_scores'].values()) < best_fid:
                best_fid = sum(eval_results['fid_scores'].values())
                n_p = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_fid': sum(eval_results['fid_scores'].values()),
                    'train_loss': train_loss
                }, self.model_ckpt)
                print("Epoch completed and model successfully saved!")
            else:
                n_p += 1
            if n_p > self.es_paiteince:
                print('Early Stopping')
                break
        print('Training Finished')
        print('best_fid', sum(eval_results['fid_scores'].values()))

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

        self.model=self.model.to(DEVICE)
        self.classifier_model=self.classifier_model.to(DEVICE)
        beta_t = [0.001 * (i + 1) for i in range(self.num_timesteps)]
        beta_t = torch.tensor(beta_t).to(DEVICE)

        checkpoint = torch.load(self.model_ckpt)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Model successfully loaded!")
        test_loader = DataLoader(
            self.test_set, batch_size=self.val_batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers)
        eval_results = self.eval_function(test_loader, self.model,self.classifier_model, beta_t,DEVICE)
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
        self.model=self.model.to(DEVICE)
        self.classifier_model = self.classifier_model.to(DEVICE)
        beta_t = [0.001 * (i + 1) for i in range(self.num_timesteps)]
        beta_t = torch.tensor(beta_t).to(DEVICE)

        checkpoint = torch.load(self.model_ckpt)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Model successfully loaded!")
        inference_loader = DataLoader(
            inference_set, batch_size=self.val_batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers)
        inference_outputs = self.eval_function(inference_loader, self.model,self.classifier_model,beta_t, DEVICE,is_inference=True)

        return inference_outputs

    def input_dataset(self, dataset,classifier_ckp=None):
        if classifier_ckp is None:
            raise ValueError("Image classifier ckpt (needed to train a good classification-based generation model) is None. Please ensure a valid ckpt is provided.")
        self.dataset=dataset
        self.train_set = dataset.train_set
        self.test_set = dataset.test_set
        self.n_classes = dataset.n_classes
        self.image_size=dataset.image_size
        self.num_timesteps = 10
        self.model = DiffusionModel(num_timesteps = self.num_timesteps,num_classes=self.n_classes)
        self.classifier_model = resnet18(num_classes=self.n_classes)
        self.classifier_model.load_state_dict(torch.load(classifier_ckp)['model_state_dict'])
        n_samples = self.train_set.class_counts
        n_samples = np.array(n_samples)
        weights = 1.0 / n_samples
        weights = weights / np.sum(weights)
        w1, w2 = weights.tolist()
        self.class_weights = torch.tensor([w1, w2], dtype=torch.float)
        self.class_labels=self.train_set.class_labels

