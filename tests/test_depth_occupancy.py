"""Regression scenes for depth occupancy: empty, left, center, right,
blocked, outside plus NaN and camera pitch invariance."""
from __future__ import annotations

import unittest

import numpy as np

from config import DEPTH_GRID_COLS, DEPTH_GRID_ROWS
from depth_detector import DepthObservation
from detector import DepthOccupancyEstimator, depth_lane_occupancy, perspective_for_pitch
from tests.depth_scenes import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    floor_plane,
    observation,
    place_box,
    run_scene,
)


class DepthOccupancyScenes(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plane = floor_plane()
        cls.upright = perspective_for_pitch(20.0)

    def test_empty_floor_creates_no_obstacle(self):
        obstacles = run_scene(self.plane, self.upright)
        self.assertEqual(obstacles, [])
        self.assertFalse(any(depth_lane_occupancy(obstacles).values()))

    def test_flat_constant_plane_creates_no_obstacle(self):
        obstacles = run_scene(np.full((FRAME_HEIGHT, FRAME_WIDTH), 3.0), self.upright)
        self.assertEqual(obstacles, [])

    def test_close_box_occupies_only_its_lane(self):
        scenes = {
            "left": (0.08, 0.25),
            "center": (0.42, 0.60),
            "right": (0.75, 0.92),
        }
        for lane, (x1, x2) in scenes.items():
            with self.subTest(lane=lane):
                scene = place_box(
                    self.plane, (0.82, 0.96), (x1, x2)
                )
                obstacles = run_scene(scene, self.upright)
                lanes = depth_lane_occupancy(obstacles)
                self.assertTrue(lanes[lane], f"{lane} lane must be occupied")
                self.assertFalse(lanes["center"] and lane != "center")
                self.assertFalse(lanes["left"] and lane != "left")
                self.assertFalse(lanes["right"] and lane != "right")

    def test_blocked_when_all_lanes_occupied(self):
        scene = self.plane.copy()
        for x1, x2 in ((0.08, 0.25), (0.42, 0.60), (0.75, 0.92)):
            scene = place_box(scene, (0.82, 0.96), (x1, x2))
        obstacles = run_scene(scene, self.upright)
        lanes = depth_lane_occupancy(obstacles)
        self.assertTrue(all(lanes.values()))

    def test_object_outside_trapezoid_is_ignored(self):
        scene = place_box(self.plane, (0.05, 0.25), (0.42, 0.60))
        obstacles = run_scene(scene, self.upright)
        self.assertEqual(obstacles, [])

    def test_nan_depth_map_does_not_crash(self):
        estimator = DepthOccupancyEstimator()
        nan_map = np.full((FRAME_HEIGHT, FRAME_WIDTH), np.nan, np.float32)
        for frame_index in range(3):
            obstacles = estimator.update(
                observation(nan_map, created_at=1000.0 + frame_index),
                FRAME_WIDTH,
                FRAME_HEIGHT,
                self.upright,
            )
        self.assertEqual(obstacles, [])

    def test_empty_depth_map_does_not_crash(self):
        estimator = DepthOccupancyEstimator()
        empty = np.zeros((0, 0), np.float32)
        obstacles = estimator.update(
            observation(empty), FRAME_WIDTH, FRAME_HEIGHT, self.upright
        )
        self.assertEqual(obstacles, [])

    def test_pitch_change_moves_mask_but_keeps_coordinates_valid(self):
        # A box between the upright zone top (0.72) and the down-looking
        # zone top (0.64): visible only when the phone is tilted down.
        scene = place_box(self.plane, (0.66, 0.70), (0.42, 0.60))
        upright_obstacles = run_scene(scene, perspective_for_pitch(0.0))
        down_obstacles = run_scene(scene, perspective_for_pitch(75.0))
        self.assertEqual(upright_obstacles, [])
        lanes = depth_lane_occupancy(down_obstacles)
        self.assertTrue(lanes["center"])
        for item in down_obstacles:
            x1, y1, x2, y2 = item.box
            self.assertTrue(0 <= x1 <= x2 <= FRAME_WIDTH)
            self.assertTrue(0 <= y1 <= y2 <= FRAME_HEIGHT)
            self.assertIn(item.lane, ("left", "center", "right"))

    def test_repeated_same_observation_is_cached(self):
        estimator = DepthOccupancyEstimator()
        scene = place_box(self.plane, (0.82, 0.96), (0.42, 0.60))
        first = estimator.update(
            observation(scene), FRAME_WIDTH, FRAME_HEIGHT, self.upright
        )
        cached = estimator.update(
            observation(scene), FRAME_WIDTH, FRAME_HEIGHT, self.upright
        )
        self.assertEqual(first, cached)


if __name__ == "__main__":
    unittest.main()
