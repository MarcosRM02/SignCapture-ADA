""" Normalizer for processing gold data. """

from src.utils.landmark import LandmarkPoint
import pandas as pd


_METADATA_COLUMNS = ['image_path', 'letter', 'original_id']

def normalize_landmarks(input_df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Normalize hand landmarks in a [-1, 1] range for each image in the
    silver directory and save the normalized landmarks to a DataFrame.

    Args:
        input_df (pd.DataFrame): The DataFrame containing the extracted landmarks.

    Returns:
        pd.DataFrame: A DataFrame containing the normalized hand landmarks and
                      preserved metadata columns present at input.
    """
    normalized_data = []
    metadata_columns = [column for column in _METADATA_COLUMNS if column in input_df.columns]

    for _, row in input_df.iterrows():
        landmarks = [LandmarkPoint(row[f'landmark{i}_x'], row[f'landmark{i}_y'], row[f'landmark{i}_z']) for i in range(21)]
        min_x = min(landmark.x for landmark in landmarks)
        max_x = max(landmark.x for landmark in landmarks)
        min_y = min(landmark.y for landmark in landmarks)
        max_y = max(landmark.y for landmark in landmarks)
        min_z = min(landmark.z for landmark in landmarks)
        max_z = max(landmark.z for landmark in landmarks)

        normalized_row = {
            column: row[column]
            for column in metadata_columns
        }
        for i, landmark in enumerate(landmarks):
            normalized_row[f'landmark{i}_x'] = (landmark.x - min_x) / (max_x - min_x) * 2 - 1 if max_x > min_x else 0
            normalized_row[f'landmark{i}_y'] = (landmark.y - min_y) / (max_y - min_y) * 2 - 1 if max_y > min_y else 0
            normalized_row[f'landmark{i}_z'] = (landmark.z - min_z) / (max_z - min_z) * 2 - 1 if max_z > min_z else 0
        normalized_data.append(normalized_row)

    normalized_df = pd.DataFrame(normalized_data)
    
    return normalized_df 
    
