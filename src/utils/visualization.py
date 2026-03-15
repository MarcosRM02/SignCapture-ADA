"""  
src/utils/visualization.py debe contener:

draw_landmarks_on_image(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray — dibuja los 21 puntos sobre la imagen con OpenCV
"""

import cv2
import numpy as np
from typing import List
from src.utils.landmark import LandmarkPoint

def draw_landmarks_on_image(image: np.ndarray, landmarks: List[LandmarkPoint]) -> np.ndarray:
    """
    Draw the 21 hand landmarks on the image using OpenCV.

    Args:
        image (np.ndarray): The image on which to draw the landmarks.
        landmarks (List[LandmarkPoint]): A list of LandmarkPoint objects containing the x, y, z coordinates of the 21 hand landmarks.

    Returns:
        np.ndarray: The image with the landmarks drawn.
    """
    h, w = image.shape[:2]
    
    # Draw connections between landmarks
    image = _draw_connections_on_image(image, landmarks)

    # Draw the landmarks as circles on the image
    for landmark in landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
    
    return image

def _draw_connections_on_image(image: np.ndarray, landmarks: List[LandmarkPoint]) -> np.ndarray:
    """
    Draw the connections between the hand landmarks on the image using OpenCV.

    Args:
        image (np.ndarray): The image on which to draw the connections.
        landmarks (List[LandmarkPoint]): A list of LandmarkPoint objects containing the x, y, z coordinates of the 21 hand landmarks.

    Returns:
        np.ndarray: The image with the connections drawn.
    """
    h, w = image.shape[:2]
    
    connections = [
        (0,1), (1,2), (2,3), (3,4),         # Thumb
        (0,5), (5,6), (6,7), (7,8),         # Index
        (0,9), (9,10), (10,11), (11,12),    # Middle
        (0,13), (13,14), (14,15), (15,16),  # Ring
        (0,17), (17,18), (18,19), (19,20)   # Pinky
    ]

    for connection in connections:
        start_idx, end_idx = connection
        start_point = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        end_point = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
        cv2.line(image, start_point, end_point, color=(0, 0, 255), thickness=2)
    
    return image