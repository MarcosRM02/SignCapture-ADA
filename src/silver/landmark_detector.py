""" Landmark detector for processing hand landmarks. """

from src.config import config
from src.utils.landmark import LandmarkPoint
from typing import List
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class LandmarkDetector:
    """ 
    A class to detect hand landmarks using MediaPipe.

    Attributes:
        mp_hands (mp.solutions.hands): The MediaPipe Hands solution.
    """
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=config.mediapipe.model_path)
        options = vision.HandLandmarkerOptions(base_options=base_options, 
                                               num_hands=config.mediapipe.max_num_hands)
        self.mp_hands = vision.HandLandmarker.create_from_options(options)

    def detect_landmarks(self, image)->List[LandmarkPoint]:
        """ 
        Detect hand landmarks in the given image.

        Args:
            image (numpy.ndarray): The input image in which to detect hand landmarks.
        Returns:
            List[LandmarkPoint]: A list of detected hand landmarks as LandmarkPoint instances.
        """
        # Convert the image to the format expected by MediaPipe
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Perform hand landmark detection
        results = self.mp_hands.detect(image)
        landmarks = []
        if results.hand_landmarks:
            for hand_landmark in results.hand_landmarks:
                for landmark in hand_landmark:
                    landmarks.append(LandmarkPoint(x=landmark.x, y=landmark.y, z=landmark.z))
        return landmarks

    def close(self):
        """ 
        Close the MediaPipe Hands solution to release resources.
        """
        self.mp_hands.close()