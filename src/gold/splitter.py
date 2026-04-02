""" Splitter for processing gold data. """

import pandas as pd
from pathlib import Path
from src.config import GeneralConfig


def split_landmarks(df: pd.DataFrame, drop_image_path: bool = True) -> tuple:
    """
    Split hand landmarks into train, validation, and test sets using stratified sampling.
    
    Args:
        df (pd.DataFrame): DataFrame containing landmark data and either 'letter'
                           or 'image_path' to infer the class label.
        drop_image_path (bool): Whether to remove image_path before returning splits.
    
    Returns:
        tuple: (train_df, val_df, test_df) - DataFrames for each split with exact
               class balance (same number of rows per letter within each split),
               preserving metadata such as 'original_id'.
    """
    config = GeneralConfig()
    
    df = df.copy()
    if 'letter' not in df.columns:
        if 'image_path' not in df.columns:
            raise ValueError("Input DataFrame must contain 'letter' or 'image_path'.")
        df['letter'] = df['image_path'].apply(_extract_letter_from_path)
    
    # Drop image_path column if present to avoid leaking source path to model input
    if drop_image_path and 'image_path' in df.columns:
        df = df.drop(columns=['image_path'])
    
    # Keep augmentation families together when possible and enforce exact class balance.
    if 'original_id' in df.columns:
        train_df, val_df, test_df = _split_by_original_id_balanced(df, config)
    else:
        train_df, val_df, test_df = _split_rows_balanced(df, config)
    
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    return train_df, val_df, test_df


def _extract_letter_from_path(image_path: str) -> str:
    """
    Extract the letter (sign character) from the image path.
    
    Args:
        image_path (str): Full path to the image file.
                         Expected format: ".../data/bronze/A/0002.jpg"
    
    Returns:
        str: The letter character (e.g., "A").
    """
    path = Path(image_path)
    # The parent directory name is the letter
    letter = path.parent.name
    return letter


def _compute_class_split_sizes(total_per_class: int, config: GeneralConfig) -> tuple[int, int, int]:
    """Compute train/val/test sizes per class from configured ratios."""
    train_size = int(total_per_class * config.train_ratio)
    val_size = int(total_per_class * config.val_ratio)
    test_size = total_per_class - train_size - val_size

    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        raise ValueError(
            "Split ratios produce an empty split. Increase samples per class or adjust ratios."
        )

    return train_size, val_size, test_size


def _split_by_original_id_balanced(df: pd.DataFrame, config: GeneralConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by original_id groups with exact class balance in each split."""
    groups_df = df[['original_id', 'letter']].drop_duplicates().reset_index(drop=True)
    class_counts = groups_df['letter'].value_counts()
    min_groups_per_class = int(class_counts.min())

    train_size, val_size, test_size = _compute_class_split_sizes(min_groups_per_class, config)

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []

    for letter in sorted(class_counts.index):
        letter_groups = groups_df[groups_df['letter'] == letter]
        selected_groups = letter_groups.sample(n=min_groups_per_class, random_state=config.seed)
        shuffled_groups = selected_groups.sample(frac=1.0, random_state=config.seed).reset_index(drop=True)

        train_ids.extend(shuffled_groups.iloc[:train_size]['original_id'].tolist())
        val_ids.extend(shuffled_groups.iloc[train_size:train_size + val_size]['original_id'].tolist())
        test_ids.extend(shuffled_groups.iloc[train_size + val_size:train_size + val_size + test_size]['original_id'].tolist())

    train_df = df[df['original_id'].isin(train_ids)]
    val_df = df[df['original_id'].isin(val_ids)]
    test_df = df[df['original_id'].isin(test_ids)]
    return train_df, val_df, test_df


def _split_rows_balanced(df: pd.DataFrame, config: GeneralConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows directly with exact class balance in each split."""
    class_counts = df['letter'].value_counts()
    min_rows_per_class = int(class_counts.min())

    train_size, val_size, test_size = _compute_class_split_sizes(min_rows_per_class, config)

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for letter in sorted(class_counts.index):
        letter_df = df[df['letter'] == letter]
        selected_rows = letter_df.sample(n=min_rows_per_class, random_state=config.seed)
        shuffled_rows = selected_rows.sample(frac=1.0, random_state=config.seed).reset_index(drop=True)

        train_parts.append(shuffled_rows.iloc[:train_size])
        val_parts.append(shuffled_rows.iloc[train_size:train_size + val_size])
        test_parts.append(shuffled_rows.iloc[train_size + val_size:train_size + val_size + test_size])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, val_df, test_df

