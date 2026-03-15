""" Pipeline for processing silver data. """

from src.config import config
from src.silver.landmark_detector import LandmarkDetector
from src.silver.landmark_extractor import extract_landmarks

def run_silver_pipeline():
    """ 
    Run the silver pipeline to extract hand landmarks from images. 
    """
    detector = LandmarkDetector()
    extract_landmarks(config.paths.bronze_dir, config.paths.silver_dir, detector)
    detector.close()

if __name__ == "__main__":
    run_silver_pipeline()