""" Module for downloading the Kaggle dataset. """

import os
import kaggle
import shutil
from src.config import config
import numpy as np

def download_kaggle_dataset():
    """ 
    Downloads the ASL Alphabet Dataset from Kaggle and saves it to the bronze directory.
    """
    # Ensure the bronze directory exists
    bronze_dir = config.paths.bronze_dir
    if not bronze_dir.exists():
        bronze_dir.mkdir(parents=True, exist_ok=True)

    # Download the dataset using Kaggle API
    kaggle.api.dataset_download_files(config.dataset.repo, path=str(bronze_dir), unzip=True)

def process_downloaded_dataset():
    """  
    Processes the downloaded dataset by restructuring it and removing unwanted images.
    This includes:
    1. Restructuring the dataset to ensure each class has its own subdirectory.
    2. Removing images containing 'rotate' in their names.
    3. Removing images to ensure exactly num_instances_per_sign images per class, spaced uniformly if needed.
    4. Renaming remaining images to a consistent format (e.g., 0001.jpg, 0002.jpg, etc.).
    """
    # Restructure the downloaded dataset
    _restructure_downloaded_dataset()

    # Remove unwanted images and ensure the correct number of instances per class
    _remove_images()

def _restructure_downloaded_dataset():
    """  
    Restructures the downloaded dataset to ensure that each class has its own subdirectory.
    """
    # Move images from the downloaded structure to the desired structure
    bronze_dir = config.paths.bronze_dir
    asl_alphabet_dir = bronze_dir / "ASL_Alphabet_Dataset"
    
    if not asl_alphabet_dir.exists():
        print("Dataset not found. Please run the download_kaggle_dataset function first.")
        return
    
    for split in ['asl_alphabet_test', 'asl_alphabet_train']:
        split_dir = asl_alphabet_dir / split
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                target_class_dir = bronze_dir / class_dir.name
                target_class_dir.mkdir(exist_ok=True)
                for img_file in class_dir.iterdir():
                    if img_file.is_file():
                        target_img_path = target_class_dir / img_file.name
                        img_file.rename(target_img_path)

    # Remove the non-useful directories
    non_useful_dirs = ['ASL_Alphabet_Dataset', 'del', 'nothing', 'space', 'J', 'Z']
    for dir_name in non_useful_dirs:
        dir_path = bronze_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            shutil.rmtree(dir_path)

def _remove_images():
    """  
    For each class subdirectory in bronze_dir, removes images to ensure exactly
    num_instances_per_sign images per class. 
    
    Removes images containing 'rotate' in the name, then removes images spaced 
    uniformly if needed, and finally renames remaining images to a consistent 
    format (e.g., 0001.jpg, 0002.jpg, etc.).
    """
    # Get configuration values
    bronze_dir = config.paths.bronze_dir
    num_instances_per_sign = config.dataset.num_instances_per_sign

    for subdir in bronze_dir.iterdir():
        if not subdir.is_dir():
            continue

        files = sorted([f for f in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, f))])

        # Remove 'rotate' images first
        files_no_rotate = []
        for file in files:
            if "rotate" in file:
                os.remove(os.path.join(subdir, file))
            else:
                files_no_rotate.append(file)

        files = files_no_rotate

        total_files = len(files)
        if total_files > num_instances_per_sign:
            # Calculate indices to delete, spaced uniformly
            num_to_delete = total_files - num_instances_per_sign
            indices_to_delete = np.linspace(0, total_files - 1, num_to_delete, dtype=int)
            indices_to_delete_set = set(indices_to_delete)
        else:
            indices_to_delete_set = set()

        # Remove selected files
        for idx, file in enumerate(files):
            if idx in indices_to_delete_set:
                os.remove(os.path.join(subdir, file))

        # Get remaining files and rename them
        remaining_files = [f for idx, f in enumerate(files) if idx not in indices_to_delete_set]
        for new_idx, file in enumerate(sorted(remaining_files), 1):
            ext = os.path.splitext(file)[1]
            new_name = f"{new_idx:04d}{ext}"
            old_path = os.path.join(subdir, file)
            new_path = os.path.join(subdir, new_name)
            if old_path != new_path:
                os.rename(old_path, new_path)
