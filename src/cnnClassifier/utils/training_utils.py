from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import os
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import torch
from cnnClassifier import logger
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from pathlib import Path

def validate_model(model,valid_loader,checkpoint_dir,epoch,best_val_accuracy,
                   device,val_loss_list,val_accuracy_list):

      error = nn.CrossEntropyLoss()

      labels_list2 = []
      predictions_list2 = []
      correct = 0
      total2 = 0
      running_loss2 = 0

      with torch.inference_mode():

        for images, labels in valid_loader:
          images, labels = images.to(device), labels.to(device)
          labels_list2.append(labels)

          if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)  # Repeat the grayscale channel 3 times

          test = (images.view(100, 3,28, 28))

          outputs = model(test)
          loss = error(outputs, labels)
          running_loss2 += loss.item() * images.size(0)

          predictions = torch.max(outputs, 1)[1].to(device)
          predictions_list2.append(predictions)
          correct += (predictions == labels).sum()

          total2 += len(labels)

        labels_cpu2 = torch.cat(labels_list2).cpu().numpy() # Move to CPU and convert to NumPy
        predictionss_cpu2 = torch.cat(predictions_list2).cpu().numpy() # Move to CPU and convert to NumPy


        validation_accuracy = accuracy_score(labels_cpu2, predictionss_cpu2)
        validation_loss = running_loss2 / total2
        logger.info("validation_loss: {},validation_accuracy: {}%".format(validation_loss, validation_accuracy))
        mlflow.log_metrics({f"epoch{epoch}_validation_accuracy": float(validation_accuracy), f"epoch{epoch}_validation_loss": float(validation_loss)})

        # Save model checkpoint if test accuracy improves
        if validation_accuracy > best_val_accuracy:
            best_val_accuracy = validation_accuracy
            checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_model.pth')
            
            torch.save(model, checkpoint_path)
            print(f'Model checkpoint saved at {checkpoint_path}')

        val_loss_list.append(validation_loss)
        val_accuracy_list.append(validation_accuracy)


def plot_loss_accuracy(train_loss_list, train_accuracy_list, val_loss_list, val_accuracy_list,num_epochs):

    epochs = range(0, num_epochs )

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_list, 'b', label='Training Loss')
    plt.plot(epochs, val_loss_list, 'r', label='val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and val Loss')
    plt.savefig('Training and val Loss.png')
    mlflow.log_artifact('Training and val Loss.png')


    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy_list, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracy_list, 'r', label='val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and val Accuracy')
    plt.savefig('Training and val Accuracy.png')
    mlflow.log_artifact('Training and val Accuracy.png')

    plt.show()


def evaluation_metrics_n_Hyperparameters(self,labels_list,predictions_list,train_loss,epoch):
      # Calculate evaluation metrics
      labels_cpu = torch.cat(labels_list).cpu().numpy() # Move to CPU and convert to NumPy
      predictionss_cpu = torch.cat(predictions_list).cpu().numpy() # Move to CPU and convert to NumPy


      accuracy = accuracy_score(labels_cpu, predictionss_cpu)
      precision = precision_score(labels_cpu, predictionss_cpu, average='macro')
      recall = recall_score(labels_cpu, predictionss_cpu, average='macro')

      # Log metrics

      mlflow.log_metric(f'epoch{epoch}_accuracy', accuracy)
      mlflow.log_metric(f'epoch{epoch}_precision', precision)
      mlflow.log_metric(f'epoch{epoch}_recall', recall)
      mlflow.log_metric(f'epoch{epoch}_loss', train_loss)

      logger.info("epoch: {}, Loss: {}, Accuracy: {}%".format(epoch, train_loss, accuracy))

      self.train_loss_list.append(train_loss)
      self.train_accuracy_list.append(accuracy)

def log_model_n_params(model,labels_list,predictions_list,params):

    labels_cpu = torch.cat(labels_list).cpu().numpy() # Move to CPU and convert to NumPy
    predictionss_cpu = torch.cat(predictions_list).cpu().numpy() # Move to CPU and convert to NumPy

    mlflow.pytorch.log_model(
    pytorch_model=model,
    artifact_path="model1",
    signature=None,
    registered_model_name="model1",)

    # Log confusion matrix (example assuming you convert it to JSON)
    cm = confusion_matrix(labels_cpu, predictionss_cpu)
    cm_json = {'confusion_matrix': cm.tolist()}  # Convert to JSON or other suitable format
    mlflow.log_param('confusion_matrix', cm_json)

    # Log parameters
    mlflow.log_params(params)

    # plot and log confusion matrix
    plt.figure(figsize=(12, 12))
    sns.heatmap(cm_json['confusion_matrix'], annot=True, cmap='coolwarm', square=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')

    print(cm_json.keys())
    print(cm_json['confusion_matrix'])



