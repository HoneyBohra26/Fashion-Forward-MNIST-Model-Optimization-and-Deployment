import os
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size,transform_train,transform_test
from torch.utils.data import Dataset, DataLoader
import torchvision
from cnnClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def download_file(self)-> str:

        try: 
            train_data=self.config.train_data
            test_data=self.config.test_data 
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {torchvision.datasets} into folder {train_data},{test_data}")

            torchvision.datasets.FashionMNIST(train_data, download=True,transform=transform_train)
            torchvision.datasets.FashionMNIST(test_data, download=True, train=False, transform=transform_test)

            logger.info(f"Downloaded data from {torchvision.datasets} into folder {train_data},{test_data}")

        except Exception as e:
            raise e
        
