""" Pipeline for running the bronze layer processes. """

from src.config import config
from src.bronze.downloader import download_kaggle_dataset, process_downloaded_dataset

def run_bronze_pipeline():
    """ 
    Runs the bronze layer pipeline, which includes:
    1. Downloading the ASL Alphabet Dataset from Kaggle.
    2. Processing the downloaded dataset by restructuring it and removing unwanted images.
    """
    download_kaggle_dataset()
    process_downloaded_dataset()

if __name__ == "__main__":
    run_bronze_pipeline()