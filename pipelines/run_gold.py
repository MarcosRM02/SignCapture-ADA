""" Pipeline for running gold-level data processing. """

import pandas as pd
from src.config import config
from src.gold.normalizer import normalize_landmarks
from src.gold.feature_engineering import add_angle_features
from src.gold.augmentor import augment_landmarks, ensure_original_id
from src.gold.splitter import split_landmarks

def run_gold_pipeline():
    """ 
    Run the gold pipeline to split, augment only train, and normalize landmarks. 
    """
    silver_df = pd.read_csv(config.paths.silver_dir / 'hand_landmarks.csv')
    silver_df = ensure_original_id(silver_df)

    train_raw_df, val_raw_df, test_raw_df = split_landmarks(silver_df, drop_image_path=False)

    train_augmented_df = augment_landmarks(input_df=train_raw_df)
    train_df = add_angle_features(normalize_landmarks(train_augmented_df))
    val_df = add_angle_features(normalize_landmarks(val_raw_df))
    test_df = add_angle_features(normalize_landmarks(test_raw_df))

    if 'image_path' in train_df.columns:
        train_df = train_df.drop(columns=['image_path'])
    if 'image_path' in val_df.columns:
        val_df = val_df.drop(columns=['image_path'])
    if 'image_path' in test_df.columns:
        test_df = test_df.drop(columns=['image_path'])

    config.paths.gold_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(config.paths.gold_dir / 'train.csv', index=False)
    val_df.to_csv(config.paths.gold_dir / 'val.csv', index=False)
    test_df.to_csv(config.paths.gold_dir / 'test.csv', index=False)

if __name__ == "__main__":
    run_gold_pipeline()