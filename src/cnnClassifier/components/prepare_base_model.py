import os
import torch
import torch.nn as nn
import torchvision
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path



class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = torchvision.models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.config.param_classes)

        self.save_model(path=self.config.base_model_path, model=self.model)
    
    @staticmethod
    def _prepare_full_model(classes,model, freeze_all, freeze_till,):
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False
        elif freeze_till is not None and freeze_till > 0:
            for param in list(model.parameters())[:-freeze_till]:
                param.requires_grad = False

        # Modify the final layer for the number of classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, classes)

        return model
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,           
            freeze_all=True,
            freeze_till=None,
            classes=self.config.param_classes,

        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
    
    @staticmethod
    def save_model(path: Path, model: nn.Module):
        torch.save(model, path)

