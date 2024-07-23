import os
from box.exceptions import BoxValueError
import yaml
from cnnClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Define data augmentation transformations

@ensure_annotations
def transform_train() :
    return (transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
]))

@ensure_annotations
def transform_test(path: Path) -> ConfigBox:

   transform =(transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
]))

   return ConfigBox(transform)

def objective(params,model,train_loader,val_loader):
    """Objective function for hyperparameter optimization."""
    learning_rate = params['learning_rate']
    num_epochs = int(params['num_epochs'])
    optimizer_name = params['optimizer']
    batch_size = int(params['batch_size'])
    error = nn.CrossEntropyLoss()

        # Initialize the optimizer based on the optimizer_name
    if optimizer_name == 'Adam':
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD':
      optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    count = 0
    # Lists for knowing classwise accuracy
    predictions_list = []
    labels_list = []

    running_loss = 0
    total = 0

    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):

      for images, labels in train_loader:

          # Transfering images and labels to GPU if available
          images, labels = images.to(device), labels.to(device)

          if images.shape[1] == 1:
           images = images.repeat(1, 3, 1, 1)  # Repeat the grayscale channel 3 times

          train = (images.view(100, 3, 28, 28))
          labels = (labels)


          # Forward pass
          outputs = model(train)
          loss = error(outputs, labels)
          running_loss += loss.item() * images.size(0)
          predictions = torch.max(outputs, 1)[1].to(device)
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
    labels_list2 = []
    predictions_list2 = []
    correct = 0
    total2 = 0
    running_loss2 = 0

    with torch.inference_mode():

      for images, labels in val_loader:
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


      val_accuracy = accuracy_score(labels_cpu2, predictionss_cpu2)
      val_loss = running_loss2 / total2

    return {'loss': val_loss, 'status': STATUS_OK}


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")






@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())

