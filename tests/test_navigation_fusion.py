"""Fusion rules: stale depth, decision source, and STOP/GO hysteresis."""
from __future__ import annotations

import unittest

from config import CLEAR_FRAMES_REQUIRED
from depth_detector import observation_is_fresh
from detector import DecisionFilter, Detection, navigate, perspective_for_pitch
from tests.depth_scenes import (
    floor_plane,
    observation,
    place_box,
    run_scene,
)


def danger(label: str = "person") -> Detection:
    return Detection(
        label=label,
        confidence=0.9,
        box=(100, 800, 200, 1000),
        lane="center",
        close=True,
        dangerous=True,
    )


class DepthNavigationFusion(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plane = floor_plane()
        cls.perspective = perspective_for_pitch(20.0)

    def depth_obstacles_center(self):
        scene = place_box(self.plane, (0.82, 0.96), (0.42, 0.60))
        return run_scene(scene, self.perspective)

    def test_freshness_boundary(self):
        current = observation(self.plane, created_at=1000.0)
        self.assertTrue(observation_is_fresh(current, now=1000.5))
        self.assertTrue(observation_is_fresh(current, now=1001.0))
        self.assertFalse(observation_is_fresh(current, now=1001.001))
        self.assertFalse(observation_is_fresh(None, now=1000.0))

    def test_depth_only_obstacle_stops_with_unknown_reason(self):
        result = navigate(
            detections=[],
            depth_obstacles=self.depth_obstacles_center(),
            depth_fresh=True,
            too_dark=False,
        )
        self.assertEqual(result.reason, "unknown_obstacle")
        self.assertEqual(result.source, "depth")
        self.assertTrue(result.lanes["center"])

    def test_depth_and_yolo_together_report_yolo_label(self):
        result = navigate(
            detections=[danger("box")],
            depth_obstacles=self.depth_obstacles_center(),
            depth_fresh=True,
            too_dark=False,
        )
        self.assertEqual(result.reason, "box")
        self.assertEqual(result.source, "depth + yolo")
        self.assertTrue(result.lanes["center"])
        self.assertEqual(result.route, "LEFT")

    def test_yolo_only_still_stops_immediately(self):
        result = navigate(
            detections=[danger("person")],
            depth_obstacles=[],
            depth_fresh=True,
            too_dark=False,
        )
        self.assertEqual(result.reason, "person")
        self.assertEqual(result.source, "yolo")
        self.assertTrue(result.lanes["center"])

    def test_stale_depth_map_is_ignored(self):
        result = navigate(
            detections=[],
            depth_obstacles=self.depth_obstacles_center(),
            depth_fresh=False,
            too_dark=False,
        )
        self.assertIsNone(result.reason)
        self.assertIsNone(result.source)
        self.assertFalse(any(result.lanes.values()))
        self.assertEqual(result.route, "STRAIGHT")

    def test_route_union_avoids_depth_occupied_lane(self):
        scene = place_box(self.plane, (0.82, 0.96), (0.75, 0.92))
        result = navigate(
            detections=[],
            depth_obstacles=run_scene(scene, self.perspective),
            depth_fresh=True,
            too_dark=False,
        )
        self.assertTrue(result.lanes["right"])
        self.assertFalse(result.lanes["center"])
        self.assertEqual(result.route, "STRAIGHT")

    def test_route_blocked_when_all_lanes_occupied(self):
        scene = self.plane.copy()
        for x1, x2 in ((0.08, 0.25), (0.42, 0.60), (0.75, 0.92)):
            scene = place_box(scene, (0.82, 0.96), (x1, x2))
        result = navigate(
            detections=[],
            depth_obstacles=run_scene(scene, self.perspective),
            depth_fresh=True,
            too_dark=False,
        )
        self.assertEqual(result.route, "BLOCKED")

    def test_dark_frame_overrides_depth(self):
        result = navigate(
            detections=[],
            depth_obstacles=self.depth_obstacles_center(),
            depth_fresh=True,
            too_dark=True,
        )
        self.assertEqual(result.reason, "camera_dark")
        self.assertIsNone(result.source)
        self.assertEqual(result.route, "BLOCKED")


class StopGoHysteresis(unittest.TestCase):
    def test_stop_is_immediate_and_go_requires_three_clean_frames(self):
        decision_filter = DecisionFilter(CLEAR_FRAMES_REQUIRED)
        self.assertEqual(decision_filter.update(None).state, "GO")

        stopped = decision_filter.update("unknown_obstacle")
        self.assertEqual(stopped.state, "STOP")
        self.assertEqual(stopped.reason, "unknown_obstacle")

        # STOP persists while any danger remains.
        self.assertEqual(decision_filter.update("unknown_obstacle").state, "STOP")

        self.assertEqual(decision_filter.update(None).state, "STOP")
        self.assertEqual(decision_filter.update(None).state, "STOP")
        resumed = decision_filter.update(None)
        self.assertEqual(resumed.state, "GO")
        self.assertIsNone(resumed.reason)
        self.assertEqual(resumed.changed_from, "STOP")

    def test_new_danger_resets_clean_counter(self):
        decision_filter = DecisionFilter(CLEAR_FRAMES_REQUIRED)
        decision_filter.update("person")
        decision_filter.update(None)
        interrupted = decision_filter.update("box")
        self.assertEqual(interrupted.state, "STOP")
        self.assertEqual(interrupted.reason, "box")
        for _ in range(CLEAR_FRAMES_REQUIRED - 1):
            self.assertEqual(decision_filter.update(None).state, "STOP")
        self.assertEqual(decision_filter.update(None).state, "GO")


if __name__ == "__main__":
    unittest.main()
