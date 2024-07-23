from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    train_data: Path
    test_data: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    param_classes: int 

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_optimizer: int
    params_learning_rate: float
    param: dict