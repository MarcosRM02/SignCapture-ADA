""" Landmark extractor for processing hand landmarks. """

from src.silver.landmark_detector import LandmarkDetector
from src.utils.landmark import LandmarkPoint
from typing import List
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm

def extract_landmarks(bronze_dir: Path, silver_dir: Path, detector: LandmarkDetector) -> pd.DataFrame:
    """ 
    Extract hand landmarks from images in the bronze directory and save them to a DataFrame.

    Args:
        bronze_dir (Path): The directory containing the raw images.
        silver_dir (Path): The directory where the extracted landmarks will be saved.
        detector (LandmarkDetector): An instance of the LandmarkDetector class for detecting hand landmarks.
    Returns:
        pd.DataFrame: A DataFrame containing the extracted hand landmarks.
    """

    data = []
    subdirs = [subdir for subdir in bronze_dir.iterdir() if subdir.is_dir()]

    for subdir in tqdm(subdirs, desc="Extracting landmarks (by subdir)"):
        image_paths = list(subdir.glob('*.jpg'))
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            landmarks = detector.detect_landmarks(image)
            if landmarks:
                row = {'image_path': str(image_path), 'letter': subdir.name}
                for i, landmark in enumerate(landmarks):
                    row[f'landmark{i}_x'] = landmark.x
                    row[f'landmark{i}_y'] = landmark.y
                    row[f'landmark{i}_z'] = landmark.z
                data.append(row)

    df = pd.DataFrame(data)
    silver_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(silver_dir / 'hand_landmarks.csv', index=False)
    return df