""" Splitter for processing gold data. """

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.config import GeneralConfig


def split_landmarks(df: pd.DataFrame) -> tuple:
    """
    Split hand landmarks into train, validation, and test sets using stratified sampling.
    
    Args:
        df (pd.DataFrame): DataFrame containing landmark data with 'image_path' column.
    
    Returns:
        tuple: (train_df, val_df, test_df) - DataFrames for each split, with 'label' column
               instead of 'image_path'.
    """
    config = GeneralConfig()
    
    # Extract label from image_path
    df = df.copy()
    df['label'] = df['image_path'].apply(_extract_label_from_path)
    
    # Drop image_path column
    df = df.drop(columns=['image_path'])
    
    # Split: train/temp (remaining)
    train_df, temp_df = train_test_split(
        df,
        test_size=(config.val_ratio + config.test_ratio),
        random_state=config.seed,
        stratify=df['label']
    )
    
    # Split: val/test from temp
    val_size = config.val_ratio / (config.val_ratio + config.test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        random_state=config.seed,
        stratify=temp_df['label']
    )
    
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    return train_df, val_df, test_df


def _extract_label_from_path(image_path: str) -> str:
    """
    Extract the label (sign character) from the image path.
    
    Args:
        image_path (str): Full path to the image file.
                         Expected format: ".../data/bronze/A/0002.jpg"
    
    Returns:
        str: The label character (e.g., "A").
    """
    path = Path(image_path)
    # The parent directory name is the label
    label = path.parent.name
    return label

