import os
import urllib.request as request
import time
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
from torchvision import transforms
from cnnClassifier import logger
from cnnClassifier.utils.training_utils import log_model_n_params,validate_model
from cnnClassifier.utils.training_utils import evaluation_metrics_n_Hyperparameters,plot_loss_accuracy
from cnnClassifier.utils.common import transform_train
from cnnClassifier.entity.config_entity import TrainingConfig

# Define the transform_train function
def transform_train():
    return transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

class Training:

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_loss_list = []
        self.train_accuracy_list = []
        self.val_loss_list = [] 
        self.val_accuracy_list = []
        self.best_val_accuracy = 0

    def train_val_split(self):
        train_set =torchvision.datasets.FashionMNIST(self.config.training_data, download=True,transform=transform_train())


        train_ratio = 0.8  # Adjust as needed
        train_size = int(train_ratio * len(train_set))
        val_size = len(train_set) - train_size

        train_dataset, val_dataset = random_split(train_set, [train_size, val_size])

        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=self.config.params_batch_size)
        self.valid_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=self.config.params_batch_size)
    
    def get_base_model(self):
        self.model = torch.load(f'{self.config.base_model_path}', map_location=self.device)

    def save_model(self):
        torch.save(self.model, self.config.trained_model_path)
    
    def train_model(self):

        error = nn.CrossEntropyLoss()
        model1 = self.model
        # Initialize the optimizer based on optimizer_name
        if self.config.params_optimizer == 0:
            optimizer = torch.optim.Adam(model1.parameters(), lr=self.config.params_learning_rate)
        elif self.config.params_optimizer == 1:
            optimizer = torch.optim.SGD(model1.parameters(), lr=self.config.params_learning_rate)

        # Lists for knowing classwise accuracy
        predictions_list = []
        labels_list = []

        running_loss = 0
        total = 0
        scheduler = StepLR(optimizer, step_size=(self.config.params_epochs)/2, gamma=0.1)
       
        # Directory to save model checkpoints
        checkpoint_dir = 'checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)

        file_path = 'checkpoints\checkpoint_model.pth'

        if os.path.exists(file_path):
            self.model = torch.load('checkpoints\checkpoint_model.pth', map_location=self.device)
            logger.info('checkpoint_model')

        for epoch in range(self.config.params_epochs):

            for images, labels in self.train_loader:

                # Transfering images and labels to GPU if available
                images, labels = images.to(self.device), labels.to(self.device)

                if images.shape[1] == 1:
                 images = images.repeat(1, 3, 1, 1)  # Repeat the grayscale channel 3 times

                train = images.view(100, 3, 28, 28)
                labels = labels


                # Forward pass
                outputs = self.model(train)
                loss = error(outputs, labels)
                running_loss += loss.item() * images.size(0)
                predictions = torch.max(outputs, 1)[1].to(self.device)
                predictions_list.append(predictions)
                labels_list.append(labels)

                # Initializing a gradient as 0 so there is no mixing of gradient among the batches
                optimizer.zero_grad()
                #Propagating the error backward
                loss.backward()
                # Optimizing the parameters
                optimizer.step()
                # Total
                total += len(labels)

            # loss calculation
            train_loss = running_loss / total

            scheduler.step()  # Update learning rate

        # validation
            valid_complete = validate_model(self.model,self.valid_loader,checkpoint_dir,epoch,self.best_val_accuracy,
                                 self.device,self.val_loss_list,self.val_accuracy_list)
            
            logger.info(f'valid_complete = {valid_complete}')

        #logging and evaluating metrics_n_Hyperparameters
            metrics_eval_complete = evaluation_metrics_n_Hyperparameters(self,labels_list,predictions_list,train_loss,epoch)
            logger.info(f'metrics_eval_complete = {metrics_eval_complete}')

        #plotting loss and accuracy
        plotting_loss_accuracy_complete = plot_loss_accuracy(self.train_loss_list, self.train_accuracy_list, 
                           self.val_loss_list, self.val_accuracy_list,self.config.params_epochs)
        logger.info(f'plotting_loss_accuracy_complete = {plotting_loss_accuracy_complete}')

        #logging the model
        logging_model_n_params_complete = log_model_n_params(self.model,labels_list,predictions_list,self.config.param)
        logger.info(f'logging_model_n_params_complete = {logging_model_n_params_complete}')

        





