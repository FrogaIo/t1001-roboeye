from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from ultralytics import YOLOWorld

from config import (
    CLASS_PROMPTS,
    CONFIDENCE_THRESHOLD,
    CONTROL_ZONE_TOP_DOWN,
    CONTROL_ZONE_TOP_UPRIGHT,
    CORRIDOR_BOTTOM_LEFT,
    CORRIDOR_BOTTOM_RIGHT,
    CORRIDOR_TOP_HALF_WIDTH_DOWN,
    CORRIDOR_TOP_HALF_WIDTH_UPRIGHT,
    DEFAULT_CAMERA_PITCH_DEGREES,
    DEPTH_EMA_ALPHA,
    DEPTH_FLOOR_PERCENTILE,
    DEPTH_FLOOR_PROFILE_ALPHA,
    DEPTH_GRID_COLS,
    DEPTH_GRID_ROWS,
    DEPTH_MIN_COMPONENT_CELLS,
    DEPTH_MIN_COMPONENT_SCORE,
    DEPTH_RESIDUAL_MADS,
    DEPTH_RESIDUAL_MIN_DELTA,
    DISPLAY_NAMES,
    INFERENCE_SIZE,
    IOU_THRESHOLD,
    MAX_DETECTIONS,
    MAX_CAMERA_PITCH_DEGREES,
    MIN_CAMERA_PITCH_DEGREES,
    MIN_CLOSE_HEIGHT_RATIO,
    MODEL_PATH,
)
from depth_detector import DepthObservation


PROMPTS = [prompt for prompts in CLASS_PROMPTS.values() for prompt in prompts]
PROMPT_TO_CLASS = {
    prompt: canonical
    for canonical, prompts in CLASS_PROMPTS.items()
    for prompt in prompts
}
LANES = ("left", "center", "right")


@dataclass(frozen=True)
class Perspective:
    pitch_degrees: float
    zone_top: float
    top_left: float
    top_right: float
    bottom_left: float = CORRIDOR_BOTTOM_LEFT
    bottom_right: float = CORRIDOR_BOTTOM_RIGHT


def perspective_for_pitch(pitch_degrees: float | None) -> Perspective:
    pitch = DEFAULT_CAMERA_PITCH_DEGREES if pitch_degrees is None else pitch_degrees
    pitch = min(MAX_CAMERA_PITCH_DEGREES, max(MIN_CAMERA_PITCH_DEGREES, pitch))
    amount = (pitch - MIN_CAMERA_PITCH_DEGREES) / max(
        MAX_CAMERA_PITCH_DEGREES - MIN_CAMERA_PITCH_DEGREES, 1
    )
    zone_top = CONTROL_ZONE_TOP_UPRIGHT + (
        CONTROL_ZONE_TOP_DOWN - CONTROL_ZONE_TOP_UPRIGHT
    ) * amount
    half_width = CORRIDOR_TOP_HALF_WIDTH_UPRIGHT + (
        CORRIDOR_TOP_HALF_WIDTH_DOWN - CORRIDOR_TOP_HALF_WIDTH_UPRIGHT
    ) * amount
    return Perspective(
        pitch_degrees=pitch,
        zone_top=zone_top,
        top_left=0.5 - half_width,
        top_right=0.5 + half_width,
    )


def corridor_edges(
    width: int, height: int, y: float, perspective: Perspective
) -> tuple[float, float]:
    """Return floor-corridor edges at y using a perspective projection."""
    horizon = height * perspective.zone_top
    depth = min(1.0, max(0.0, (y - horizon) / max(height - horizon, 1)))
    left_ratio = perspective.top_left + (perspective.bottom_left - perspective.top_left) * depth
    right_ratio = perspective.top_right + (perspective.bottom_right - perspective.top_right) * depth
    return left_ratio * width, right_ratio * width


def projected_lane(
    width: int, height: int, x: float, y: float, perspective: Perspective
) -> str:
    left, right = corridor_edges(width, height, y, perspective)
    if x < left or x > right:
        return "outside"
    position = (x - left) / max(right - left, 1)
    if position < 1 / 3:
        return "left"
    if position < 2 / 3:
        return "center"
    return "right"


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]
    lane: str
    close: bool
    dangerous: bool


@dataclass(frozen=True)
class Decision:
    state: str
    reason: str | None
    changed_from: str | None


class DecisionFilter:
    """Stop immediately; resume only after several consecutive clear frames."""

    def __init__(self, clear_frames_required: int) -> None:
        self.state = "GO"
        self.clear_frames = 0
        self.clear_frames_required = clear_frames_required
        self.reason: str | None = None

    def update(self, danger_reason: str | None) -> Decision:
        previous = self.state
        if danger_reason:
            self.state = "STOP"
            self.reason = danger_reason
            self.clear_frames = 0
        elif self.state == "STOP":
            self.clear_frames += 1
            if self.clear_frames >= self.clear_frames_required:
                self.state = "GO"
                self.reason = None
                self.clear_frames = 0

        return Decision(
            state=self.state,
            reason=self.reason,
            changed_from=previous if previous != self.state else None,
        )


class ObjectDetector:
    def __init__(self) -> None:
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = YOLOWorld(str(MODEL_PATH))
        self.model.set_classes(PROMPTS)

    def detect(
        self, frame: np.ndarray, perspective: Perspective | None = None
    ) -> list[Detection]:
        perspective = perspective or perspective_for_pitch(None)
        height, width = frame.shape[:2]
        result = self.model.predict(
            frame,
            imgsz=INFERENCE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            agnostic_nms=True,
            max_det=MAX_DETECTIONS,
            device=self.device,
            verbose=False,
        )[0]

        detections: list[Detection] = []
        if result.boxes is None:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for raw_box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = (int(value) for value in raw_box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)

            canonical_label = PROMPT_TO_CLASS[result.names[class_id]]
            box_width = x2 - x1
            box_height = y2 - y1
            # YOLO-World occasionally interprets a wide strip along the top edge
            # as a display cube. It cannot be a nearby cube and only clutters UI.
            if (
                canonical_label == "box"
                and y1 == 0
                and y2 < height * perspective.zone_top
                and box_width > box_height * 2.2
            ):
                continue

            ground_x = (x1 + x2) / 2
            lane = projected_lane(width, height, ground_x, y2, perspective)
            box_height_ratio = (y2 - y1) / max(height, 1)
            close = (
                lane != "outside"
                and y2 >= height * perspective.zone_top
                and box_height_ratio >= MIN_CLOSE_HEIGHT_RATIO
            )
            detections.append(
                Detection(
                    label=canonical_label,
                    confidence=float(confidence),
                    box=(x1, y1, x2, y2),
                    lane=lane,
                    close=close,
                    dangerous=close and lane == "center",
                )
            )

        return detections


def lane_occupancy(detections: list[Detection]) -> dict[str, bool]:
    return {
        lane: any(item.close and item.lane == lane for item in detections)
        for lane in LANES
    }


@dataclass(frozen=True)
class DepthObstacle:
    """A connected blob closer than the floor plane of its image row.

    score is a normalized depth residual in MAD units, not metres."""

    lane: str
    score: float
    area_cells: int
    box: tuple[int, int, int, int]
    ground: tuple[int, int]


class DepthOccupancyEstimator:
    """Derives obstacle blobs in the perspective corridor from relative depth.

    The floor background of each corridor row is estimated with a low
    percentile across the corridor width and stabilized over time, so any
    surface noticeably closer than that plane becomes an obstacle. Higher
    depth values mean closer surfaces (verified in depth_smoke.py)."""

    def __init__(self) -> None:
        self._ema: np.ndarray | None = None
        self._floor_profile: np.ndarray | None = None
        self._last_created_at: float | None = None
        self._last_obstacles: list[DepthObstacle] = []

    def update(
        self,
        observation: DepthObservation,
        width: int,
        height: int,
        perspective: Perspective,
    ) -> list[DepthObstacle]:
        if observation.created_at == self._last_created_at:
            return self._last_obstacles
        self._last_created_at = observation.created_at

        if observation.depth.size == 0:
            self._last_obstacles = []
            return self._last_obstacles

        grid = cv2.resize(
            observation.depth,
            (DEPTH_GRID_COLS, DEPTH_GRID_ROWS),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        grid = self._sanitize(grid)
        if self._ema is None or self._ema.shape != grid.shape:
            self._ema = grid.copy()
        else:
            self._ema = DEPTH_EMA_ALPHA * grid + (1.0 - DEPTH_EMA_ALPHA) * self._ema

        trapezoid = np.zeros((DEPTH_GRID_ROWS, DEPTH_GRID_COLS), dtype=bool)
        row_bounds: list[tuple[int, int, int]] = []
        for row in range(DEPTH_GRID_ROWS):
            y = (row + 0.5) / DEPTH_GRID_ROWS * height
            if y < height * perspective.zone_top:
                continue
            left, right = corridor_edges(width, height, y, perspective)
            col1 = int(np.clip(left / width * DEPTH_GRID_COLS, 0, DEPTH_GRID_COLS))
            col2 = int(np.clip(right / width * DEPTH_GRID_COLS, 0, DEPTH_GRID_COLS))
            if col2 <= col1:
                continue
            trapezoid[row, col1:col2] = True
            row_bounds.append((row, col1, col2))
        if not row_bounds:
            self._last_obstacles = []
            return self._last_obstacles

        profile = np.full(DEPTH_GRID_ROWS, np.nan, dtype=np.float32)
        for row, col1, col2 in row_bounds:
            strip = self._ema[row, col1:col2]
            profile[row] = np.percentile(strip, DEPTH_FLOOR_PERCENTILE)
        if self._floor_profile is None or self._floor_profile.shape != profile.shape:
            self._floor_profile = profile
        else:
            self._floor_profile = (
                DEPTH_FLOOR_PROFILE_ALPHA * profile
                + (1.0 - DEPTH_FLOOR_PROFILE_ALPHA) * self._floor_profile
            )

        residual = self._ema - self._floor_profile[:, None]
        values = residual[trapezoid]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            self._last_obstacles = []
            return self._last_obstacles
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        delta = max(
            DEPTH_RESIDUAL_MIN_DELTA,
            DEPTH_RESIDUAL_MADS * 1.4826 * mad,
        )
        denom = max(1.4826 * mad, 1e-6)
        normalized = (residual - median) / denom

        mask = trapezoid & np.isfinite(residual) & (residual - median > delta)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        obstacles: list[DepthObstacle] = []
        labels = int(mask.max())
        if labels:
            num_labels, components = cv2.connectedComponents(mask, connectivity=8)
            for label in range(1, num_labels):
                ys, xs = np.where(components == label)
                if ys.size < DEPTH_MIN_COMPONENT_CELLS:
                    continue
                score = float(np.mean(normalized[ys, xs]))
                if score < DEPTH_MIN_COMPONENT_SCORE:
                    continue
                bottom = int(ys.max())
                ground_x = float(np.mean(xs[ys == bottom]))
                x1 = int(xs.min() / DEPTH_GRID_COLS * width)
                x2 = int((xs.max() + 1) / DEPTH_GRID_COLS * width)
                y1 = int(ys.min() / DEPTH_GRID_ROWS * height)
                y2 = int((bottom + 1) / DEPTH_GRID_ROWS * height)
                ground = (
                    int(ground_x / DEPTH_GRID_COLS * width),
                    int((bottom + 0.5) / DEPTH_GRID_ROWS * height),
                )
                lane = projected_lane(width, height, ground[0], ground[1], perspective)
                if lane == "outside":
                    continue
                obstacles.append(
                    DepthObstacle(
                        lane=lane,
                        score=round(score, 2),
                        area_cells=int(ys.size),
                        box=(x1, y1, x2, y2),
                        ground=ground,
                    )
                )
        self._last_obstacles = obstacles
        return self._last_obstacles

    @staticmethod
    def _sanitize(grid: np.ndarray) -> np.ndarray:
        finite = np.isfinite(grid)
        if finite.all():
            return grid
        fill = float(np.nanmedian(grid[finite])) if finite.any() else 0.0
        grid = grid.copy()
        grid[~finite] = fill
        return grid


def depth_lane_occupancy(obstacles: list[DepthObstacle]) -> dict[str, bool]:
    return {lane: any(item.lane == lane for item in obstacles) for lane in LANES}


def depth_lane_scores(obstacles: list[DepthObstacle]) -> dict[str, float]:
    return {
        lane: max((item.score for item in obstacles if item.lane == lane), default=0.0)
        for lane in LANES
    }


def fuse_lane_occupancy(
    yolo_lanes: dict[str, bool], depth_lanes: dict[str, bool]
) -> dict[str, bool]:
    """Depth occupancy decides safety, YOLO only adds more occupied lanes."""
    return {
        lane: yolo_lanes.get(lane, False) or depth_lanes.get(lane, False)
        for lane in LANES
    }


@dataclass(frozen=True)
class NavigationDecision:
    lanes: dict[str, bool]
    route: str
    reason: str | None
    source: str | None


def navigate(
    detections: list[Detection],
    depth_obstacles: list[DepthObstacle],
    depth_fresh: bool,
    too_dark: bool,
) -> NavigationDecision:
    """Fuse YOLO labels and depth occupancy into one navigation decision.

    Safety comes from depth occupancy, the human-readable reason from YOLO.
    A stale or missing depth map silently degrades to the YOLO-only mode."""
    yolo_lanes = lane_occupancy(detections)
    depth_lanes = (
        depth_lane_occupancy(depth_obstacles) if depth_fresh and not too_dark else {}
    )
    lanes = fuse_lane_occupancy(yolo_lanes, depth_lanes)
    route = "BLOCKED" if too_dark else choose_route(lanes)

    dangerous = [item for item in detections if item.dangerous]
    yolo_reason = dangerous[0].label if dangerous else None
    depth_center = depth_lanes.get("center", False)
    if too_dark:
        reason, source = "camera_dark", None
    elif depth_center and yolo_reason:
        reason, source = yolo_reason, "depth + yolo"
    elif depth_center:
        reason, source = "unknown_obstacle", "depth"
    elif yolo_reason:
        reason, source = yolo_reason, "yolo"
    else:
        reason, source = None, None
    return NavigationDecision(lanes=lanes, route=route, reason=reason, source=source)


def choose_route(lanes: dict[str, bool]) -> str:
    if not lanes["center"]:
        return "STRAIGHT"
    if not lanes["left"]:
        return "LEFT"
    if not lanes["right"]:
        return "RIGHT"
    return "BLOCKED"


def draw_result(
    frame: np.ndarray,
    detections: list[Detection],
    decision: Decision,
    lanes: dict[str, bool],
    route: str,
    perspective: Perspective | None = None,
) -> np.ndarray:
    perspective = perspective or perspective_for_pitch(None)
    height, width = frame.shape[:2]
    output = frame.copy()
    zone_top = int(height * perspective.zone_top)
    top_left, top_right = corridor_edges(width, height, zone_top, perspective)
    bottom_left, bottom_right = corridor_edges(width, height, height, perspective)
    top_edges = np.linspace(top_left, top_right, 4).astype(int)
    bottom_edges = np.linspace(bottom_left, bottom_right, 4).astype(int)

    overlay = output.copy()
    for index, lane in enumerate(LANES):
        color = (30, 50, 255) if lanes[lane] else (50, 255, 100)
        polygon = np.array(
            [
                [top_edges[index], zone_top],
                [top_edges[index + 1], zone_top],
                [bottom_edges[index + 1], height - 1],
                [bottom_edges[index], height - 1],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(overlay, [polygon], color)
        cv2.polylines(output, [polygon], True, color, 2, cv2.LINE_AA)
        cv2.putText(
            output,
            lane.upper(),
            (bottom_edges[index] + 9, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, 0.06, output, 0.94, 0, output)

    for item in detections:
        x1, y1, x2, y2 = item.box
        color = (30, 50, 255) if item.dangerous else (50, 255, 100)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)

        label = f"{DISPLAY_NAMES[item.label]} {item.confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2
        )
        label_top = max(0, y1 - text_height - 12)
        cv2.rectangle(
            output,
            (x1, label_top),
            (min(width - 1, x1 + text_width + 12), y1),
            color,
            -1,
        )
        cv2.putText(
            output,
            label,
            (x1 + 6, y1 - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    if decision.state == "STOP":
        band_width = max(18, int(width * 0.045))
        red_overlay = output.copy()
        cv2.rectangle(red_overlay, (0, 0), (band_width, height), (0, 0, 255), -1)
        cv2.rectangle(red_overlay, (width - band_width, 0), (width, height), (0, 0, 255), -1)
        cv2.addWeighted(red_overlay, 0.72, output, 0.28, 0, output)

    route_targets = {
        "LEFT": (int((top_edges[0] + top_edges[1]) / 2), int(height * 0.53)),
        "STRAIGHT": (int(width * 0.50), int(height * 0.54)),
        "RIGHT": (int((top_edges[2] + top_edges[3]) / 2), int(height * 0.53)),
    }
    if route in route_targets:
        cv2.arrowedLine(
            output,
            (width // 2, height - 18),
            route_targets[route],
            (255, 210, 40),
            7,
            cv2.LINE_AA,
            tipLength=0.20,
        )
    else:
        cv2.line(output, (width // 2 - 35, height - 75), (width // 2 + 35, height - 15), (0, 0, 255), 8)
        cv2.line(output, (width // 2 + 35, height - 75), (width // 2 - 35, height - 15), (0, 0, 255), 8)

    status_color = (30, 50, 255) if decision.state == "STOP" else (50, 255, 100)
    display_reason = DISPLAY_NAMES.get(decision.reason, "CLEAR")
    status = f"ROBOT: {decision.state} | {display_reason} | ROUTE: {route} | OBJECTS: {len(detections)}"
    cv2.rectangle(output, (0, 0), (width, 48), (0, 0, 0), -1)
    cv2.putText(output, status, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2, cv2.LINE_AA)
    return output
