"""Feature engineering for processing gold data."""

from __future__ import annotations

import numpy as np
import pandas as pd


_FALANGE_ANGLE_SPECS: list[tuple[str, int, int, int]] = [
    ("angle_thumb_1_2_3", 1, 2, 3),
    ("angle_thumb_2_3_4", 2, 3, 4),
    ("angle_index_5_6_7", 5, 6, 7),
    ("angle_index_6_7_8", 6, 7, 8),
    ("angle_middle_9_10_11", 9, 10, 11),
    ("angle_middle_10_11_12", 10, 11, 12),
    ("angle_ring_13_14_15", 13, 14, 15),
    ("angle_ring_14_15_16", 14, 15, 16),
    ("angle_pinky_17_18_19", 17, 18, 19),
    ("angle_pinky_18_19_20", 18, 19, 20),
]


def add_angle_features(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add angle-based features to a normalized landmarks DataFrame.

    Features added:
    - Angles between phalanxes.
    - Angles between adjacent fingers.

    Args:
        input_df (pd.DataFrame): DataFrame containing normalized landmarks.

    Returns:
        pd.DataFrame: DataFrame with the original columns plus angle features.
    """
    output_df = input_df.copy()
    landmarks = _build_landmark_tensor(output_df)

    # Angles between phalanxes
    for feature_name, idx_a, idx_b, idx_c in _FALANGE_ANGLE_SPECS:
        output_df[feature_name] = _compute_angle_degrees(
            landmarks[:, idx_a, :],
            landmarks[:, idx_b, :],
            landmarks[:, idx_c, :],
        )

    # Angles between adjacent fingers
    midpoint_59 = (landmarks[:, 5, :] + landmarks[:, 9, :]) / 2.0
    midpoint_913 = (landmarks[:, 9, :] + landmarks[:, 13, :]) / 2.0
    midpoint_1317 = (landmarks[:, 13, :] + landmarks[:, 17, :]) / 2.0

    output_df["angle_thumb_index_1_0_5"] = _compute_angle_degrees(
        landmarks[:, 1, :],
        landmarks[:, 0, :],
        landmarks[:, 5, :],
    )
    output_df["angle_index_middle_6_m59_10"] = _compute_angle_degrees(
        landmarks[:, 6, :],
        midpoint_59,
        landmarks[:, 10, :],
    )
    output_df["angle_middle_ring_10_m913_14"] = _compute_angle_degrees(
        landmarks[:, 10, :],
        midpoint_913,
        landmarks[:, 14, :],
    )
    output_df["angle_ring_pinky_14_m1317_18"] = _compute_angle_degrees(
        landmarks[:, 14, :],
        midpoint_1317,
        landmarks[:, 18, :],
    )

    return output_df


def _build_landmark_tensor(df: pd.DataFrame) -> np.ndarray:
    """Build a tensor with shape [n_samples, 21, 3] from landmark columns."""
    per_landmark_coords: list[np.ndarray] = []
    for idx in range(21):
        cols = [f"landmark{idx}_x", f"landmark{idx}_y", f"landmark{idx}_z"]
        per_landmark_coords.append(df[cols].to_numpy(dtype=np.float64, copy=False))
    return np.stack(per_landmark_coords, axis=1)


def _compute_angle_degrees(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> np.ndarray:
    """
    Compute angle ABC in degrees for batches of 3D points.

    Args:
        point_a (np.ndarray): Array [n_samples, 3] with point A.
        point_b (np.ndarray): Array [n_samples, 3] with point B (angle vertex).
        point_c (np.ndarray): Array [n_samples, 3] with point C.

    Returns:
        np.ndarray: Angles in degrees for each sample.
    """
    ba = point_a - point_b
    bc = point_c - point_b

    dot = np.einsum("ij,ij->i", ba, bc)
    norm_ba = np.linalg.norm(ba, axis=1)
    norm_bc = np.linalg.norm(bc, axis=1)
    denom = norm_ba * norm_bc

    cos_values = np.ones_like(dot)
    valid_mask = denom > 1e-12
    cos_values[valid_mask] = np.clip(dot[valid_mask] / denom[valid_mask], -1.0, 1.0)

    return np.degrees(np.arccos(cos_values))