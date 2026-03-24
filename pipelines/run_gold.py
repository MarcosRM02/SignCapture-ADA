""" Pipeline for running gold-level data processing. """

from src.config import config
from src.gold.normalizer import normalize_landmarks
from src.gold.augmentor import augment_landmarks
from src.gold.splitter import split_landmarks

def run_gold_pipeline():
    """ 
    Run the gold pipeline to normalize and augment hand landmarks. 
    """
    augmented_df = augment_landmarks(config.paths.silver_dir)
    normalized_df = normalize_landmarks(augmented_df)
    train_df, val_df, test_df = split_landmarks(normalized_df)
    train_df.to_csv(config.paths.gold_dir / 'train.csv', index=False)
    val_df.to_csv(config.paths.gold_dir / 'val.csv', index=False)
    test_df.to_csv(config.paths.gold_dir / 'test.csv', index=False)

if __name__ == "__main__":
    run_gold_pipeline()