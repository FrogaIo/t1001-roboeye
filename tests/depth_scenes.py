"""Synthetic relative-depth scenes for regression tests.

All scenes are pure numpy: no model download, no YOLO, no MPS required.
Higher depth value = closer surface, matching the verified model sign.
"""
from __future__ import annotations

import numpy as np

from depth_detector import DepthObservation

FRAME_WIDTH = 640
FRAME_HEIGHT = 1138
GRID_NOISE_STD = 0.02


def floor_plane(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gradient = np.linspace(2.0, 2.6, FRAME_HEIGHT)[:, None]
    noise = rng.normal(0.0, GRID_NOISE_STD, (FRAME_HEIGHT, FRAME_WIDTH))
    return (gradient + noise).astype(np.float32)


def place_box(
    depth: np.ndarray,
    row_range: tuple[float, float],
    col_range: tuple[float, float],
    closeness: float = 3.0,
) -> np.ndarray:
    scene = depth.copy()
    y1, y2 = (int(value * FRAME_HEIGHT) for value in row_range)
    x1, x2 = (int(value * FRAME_WIDTH) for value in col_range)
    scene[y1:y2, x1:x2] += closeness
    return scene


def observation(
    depth: np.ndarray, created_at: float = 1000.0
) -> DepthObservation:
    return DepthObservation(
        depth=depth.astype(np.float32), created_at=created_at, inference_ms=1.0
    )


def run_scene(
    depth: np.ndarray, perspective, frames: int = 3
) -> list:
    from detector import DepthOccupancyEstimator

    estimator = DepthOccupancyEstimator()
    obstacles: list = []
    for frame_index in range(frames):
        current = observation(depth, created_at=1000.0 + frame_index)
        obstacles = estimator.update(
            current, FRAME_WIDTH, FRAME_HEIGHT, perspective
        )
    return obstacles
