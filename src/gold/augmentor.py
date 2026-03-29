""" Augmentor for processing gold data. """

import math
import random
import pandas as pd
from pathlib import Path
from src.config import config
from src.config import AugmentationConfig
from src.utils.landmark import LandmarkPoint


def ensure_original_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the input DataFrame contains an original_id column.

    Args:
        df (pd.DataFrame): Input landmarks DataFrame.

    Returns:
        pd.DataFrame: DataFrame with original_id column.
    """
    if 'original_id' in df.columns:
        return df.copy()

    result_df = df.copy()
    result_df['original_id'] = [
        f"{row['letter']}_{idx}"
        for idx, (_, row) in enumerate(result_df.iterrows())
    ]
    return result_df


def augment_landmarks(silver_dir: Path | None = None, input_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Apply data augmentation to hand landmarks including rotation, zoom, and horizontal flip.
    Only uses num_images_per_sign random samples per letter (discards the rest).
    
    Args:
        silver_dir (Path | None): Directory containing the extracted landmarks.
                                  Used only when input_df is not provided.
        input_df (pd.DataFrame | None): Optional preloaded landmarks DataFrame.
    
    Returns:
        pd.DataFrame: A DataFrame containing both original and augmented hand landmarks.
                      Includes original_id to trace augmented samples back to the
                      same source observation.
    """
    if input_df is None:
        if silver_dir is None:
            raise ValueError("Either silver_dir or input_df must be provided.")
        df = pd.read_csv(silver_dir / 'hand_landmarks.csv')
    else:
        df = input_df.copy()

    df = ensure_original_id(df)
    
    # Set random seed for reproducibility
    random.seed(config.general.seed)
    
    augmented_data = []
    
    # Process each letter separately
    for letter in df['letter'].unique():
        letter_df = df[df['letter'] == letter]
        
        # Select only num_images_per_sign random rows (or all if less than that)
        num_to_use = min(config.augmentation.num_images_per_sign, len(letter_df))
        selected_rows = letter_df.sample(n=num_to_use, random_state=config.general.seed)
        
        # Preserve deterministic order: each original sample followed by its augmentations
        for _, row in selected_rows.iterrows():
            original_id = row['original_id']

            original_row = row.to_dict()
            original_row['original_id'] = original_id
            augmented_data.append(original_row)

            for _ in range(config.augmentation.num_augmentations):
                augmented_row = _apply_augmentations(
                    row=row,
                    config=config.augmentation,
                    original_id=original_id,
                )
                augmented_data.append(augmented_row)
    
    augmented_df = pd.DataFrame(augmented_data)
    
    # Print summary
    print(f"Augmentation summary:")
    for letter in df['letter'].unique():
        available = len(df[df['letter'] == letter])
        used = min(config.augmentation.num_images_per_sign, available)
        discarded = available - used
        augmented = used * config.augmentation.num_augmentations
        total = used + augmented
        print(f"  Letter {letter}: {available} available → {used} selected, {discarded} discarded → {augmented} augmented → {total} total")
    
    print(f"  Final dataset size: {len(augmented_df)}")
    
    return augmented_df


def _apply_augmentations(row: pd.Series, config: AugmentationConfig, original_id: str) -> dict:
    """
    Apply random augmentations to a single row of landmarks.
    
    Args:
        row (pd.Series): A row containing landmark data.
        config (AugmentationConfig): Augmentation configuration.
        original_id (str): Lineage identifier of the source row.
    
    Returns:
        dict: A dictionary with augmented landmark values.
    """
    # Extract landmarks from row
    landmarks = []
    for i in range(21):
        landmarks.append(LandmarkPoint(
            x=row[f'landmark{i}_x'],
            y=row[f'landmark{i}_y'],
            z=row[f'landmark{i}_z']
        ))
    
    # Calculate center of hand (centroid)
    center_x = sum(lm.x for lm in landmarks) / len(landmarks)
    center_y = sum(lm.y for lm in landmarks) / len(landmarks)
    
    # Apply random rotation
    rotation_angle = random.uniform(-config.rotation_range, config.rotation_range)
    landmarks = _rotate_landmarks(landmarks, rotation_angle, center_x, center_y)
    
    # Apply random zoom
    zoom_factor = random.uniform(1 - config.zoom_range, 1 + config.zoom_range)
    landmarks = _zoom_landmarks(landmarks, zoom_factor, center_x, center_y)
    
    # Apply horizontal flip with probability
    if config.horizontal_flip and random.random() > 0.5:
        landmarks = _flip_landmarks_horizontal(landmarks, center_x)
    
    # Build augmented row (preserve letter field)
    augmented_row = {
        'image_path': row['image_path'],
        'letter': row['letter'],
        'original_id': original_id,
    }
    for i, landmark in enumerate(landmarks):
        augmented_row[f'landmark{i}_x'] = landmark.x
        augmented_row[f'landmark{i}_y'] = landmark.y
        augmented_row[f'landmark{i}_z'] = landmark.z
    
    return augmented_row


def _rotate_landmarks(landmarks: list, angle_degrees: float, center_x: float, center_y: float) -> list:
    """
    Rotate landmarks around a center point.
    
    Args:
        landmarks (list): List of LandmarkPoint objects.
        angle_degrees (float): Rotation angle in degrees.
        center_x (float): X coordinate of rotation center.
        center_y (float): Y coordinate of rotation center.
    
    Returns:
        list: List of rotated LandmarkPoint objects.
    """
    angle_radians = math.radians(angle_degrees)
    cos_angle = math.cos(angle_radians)
    sin_angle = math.sin(angle_radians)
    
    rotated_landmarks = []
    for landmark in landmarks:
        # Translate to origin
        x = landmark.x - center_x
        y = landmark.y - center_y
        
        # Apply rotation
        rotated_x = x * cos_angle - y * sin_angle
        rotated_y = x * sin_angle + y * cos_angle
        
        # Translate back
        rotated_x += center_x
        rotated_y += center_y
        
        rotated_landmarks.append(LandmarkPoint(
            x=rotated_x,
            y=rotated_y,
            z=landmark.z
        ))
    
    return rotated_landmarks


def _zoom_landmarks(landmarks: list, zoom_factor: float, center_x: float, center_y: float) -> list:
    """
    Scale landmarks by a zoom factor.
    
    Args:
        landmarks (list): List of LandmarkPoint objects.
        zoom_factor (float): Scaling factor (>1 for zoom in, <1 for zoom out).
        center_x (float): X coordinate of zoom center.
        center_y (float): Y coordinate of zoom center.
    
    Returns:
        list: List of scaled LandmarkPoint objects.
    """
    zoomed_landmarks = []
    for landmark in landmarks:
        # Translate to origin
        x = landmark.x - center_x
        y = landmark.y - center_y
        
        # Apply zoom
        zoomed_x = x * zoom_factor + center_x
        zoomed_y = y * zoom_factor + center_y
        
        zoomed_landmarks.append(LandmarkPoint(
            x=zoomed_x,
            y=zoomed_y,
            z=landmark.z
        ))
    
    return zoomed_landmarks


def _flip_landmarks_horizontal(landmarks: list, center_x: float) -> list:
    """
    Flip landmarks horizontally (mirror across a vertical axis).
    
    Args:
        landmarks (list): List of LandmarkPoint objects.
        center_x (float): X coordinate of the flip axis.
    
    Returns:
        list: List of horizontally flipped LandmarkPoint objects.
    """
    flipped_landmarks = []
    for landmark in landmarks:
        # Mirror across the vertical axis at center_x
        flipped_x = 2 * center_x - landmark.x
        
        flipped_landmarks.append(LandmarkPoint(
            x=flipped_x,
            y=landmark.y,
            z=landmark.z
        ))
    
    return flipped_landmarks

