""" Module for handling file I/O operations """

from pathlib import Path
import yaml
import cv2
import pandas as pd
import numpy as np

def load_yaml(path: Path) -> dict:
    """
    Load a YAML file and return its contents as a dictionary.
    
    Args:
        path (Path): The path to the YAML file.

    Returns:
        dict: The contents of the YAML file as a dictionary.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def save_parquet(df: pd.DataFrame, path: Path):
    """
    Save a DataFrame to a parquet file, creating the directory if it doesn't exist.
    
    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (Path): The path to the parquet file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def load_parquet(path: Path) -> pd.DataFrame:
    """
    Load a parquet file and return it as a DataFrame.

    Args:
        path (Path): The path to the parquet file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_parquet(path)

def ensure_dir(path: Path):
    """ 
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path (Path): The path to the directory.
    """
    path.mkdir(parents=True, exist_ok=True)

def load_image(path: Path) -> np.ndarray:
    """
    Load an image from the specified path and transform into RGB.

    Args:
        path (Path): The path to the image file.
    
    Returns:
        np.ndarray: The loaded image in RGB format.
    """
    image = cv2.imread(str(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
