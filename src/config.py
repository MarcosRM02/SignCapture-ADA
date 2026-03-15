import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass
from src.utils.io import load_yaml

@dataclass
class PathsConfig:
    """ 
    Configuration class for managing file paths in the SignCapture project.

    Attributes:
        root_dir (Path): The root directory for the project.
        data_dir (Path): The directory for storing data files.
        bronze_dir (Path): The directory for storing bronze-level data.
        silver_dir (Path): The directory for storing silver-level data.
        gold_dir (Path): The directory for storing gold-level data.
    """
    root_dir: Path
    data_dir: Path
    bronze_dir: Path
    silver_dir: Path
    gold_dir: Path

    def __init__(self):
        _env_path = Path(__file__).resolve().parents[1] / '.env'
        load_dotenv(dotenv_path=_env_path)

        self.root_dir = Path(os.getenv('SIGNCAPTURE_ROOT'))
        self.data_dir = self.root_dir / 'data'
        self.bronze_dir = self.data_dir / 'bronze'
        self.silver_dir = self.data_dir / 'silver'
        self.gold_dir = self.data_dir / 'gold'


@dataclass
class MediaPipeConfig:
    """  
    Configuration class for MediaPipe settings in the SignCapture project.

    Attributes:
        max_num_hands (int): The maximum number of hands to detect.
        min_detection_confidence (float): The minimum confidence value for hand detection.
    """
    max_num_hands: int
    min_detection_confidence: float

    def __init__(self):
        _settings_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
        settings = load_yaml(_settings_path)

        self.max_num_hands = settings['mediapipe']['max_num_hands']
        self.min_detection_confidence = settings['mediapipe']['min_detection_confidence']

@dataclass
class GeneralConfig:
    """  
    Configuration class for general settings in the SignCapture project.

    Attributes:         
        seed (int): The random seed for reproducibility.
        train_ratio (float): The ratio of the dataset to be used for training.
        val_ratio (float): The ratio of the dataset to be used for validation.
        test_ratio (float): The ratio of the dataset to be used
    """
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float

    def __init__(self):
        _settings_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
        settings = load_yaml(_settings_path)
    
        self.seed = settings['general']['seed']
        self.train_ratio = settings['general']['split_ratios']['train']
        self.val_ratio = settings['general']['split_ratios']['val']
        self.test_ratio = settings['general']['split_ratios']['test']

        if not self._validate_ratios():
            self.train_ratio = 0.7
            self.val_ratio = 0.15
            self.test_ratio = 0.15

    def _validate_ratios(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not abs(total - 1.0) < 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")

@dataclass
class DatasetConfig:
    """  
    Configuration class for dataset settings in the SignCapture project.

    Attributes:
        repo (str): The Kaggle repository identifier for the dataset.
        num_instances_per_sign (int): The number of instances to keep per sign.
    """
    repo: str
    num_instances_per_sign: int

    def __init__(self):
        _settings_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
        settings = load_yaml(_settings_path)

        self.repo = settings['dataset']['repo']
        self.num_instances_per_sign = settings['dataset']['num_instances_per_sign']

@dataclass
class Config:
    """  
    Main configuration class that aggregates all other configuration classes for the SignCapture project.

    Attributes:
        paths (PathsConfig): An instance of PathsConfig containing file path configurations.
        mediapipe (MediaPipeConfig): An instance of MediaPipeConfig containing MediaPipe settings.
        general (GeneralConfig): An instance of GeneralConfig containing general settings.
        dataset (DatasetConfig): An instance of DatasetConfig containing dataset settings.
    """
    paths: PathsConfig
    mediapipe: MediaPipeConfig
    general: GeneralConfig
    dataset: DatasetConfig

    def __init__(self):
        self.paths = PathsConfig()
        self.mediapipe = MediaPipeConfig()
        self.general = GeneralConfig()
        self.dataset = DatasetConfig()

config = Config()