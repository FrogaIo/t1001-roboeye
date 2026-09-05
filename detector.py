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
    DISPLAY_NAMES,
    INFERENCE_SIZE,
    IOU_THRESHOLD,
    MAX_DETECTIONS,
    MAX_CAMERA_PITCH_DEGREES,
    MIN_CAMERA_PITCH_DEGREES,
    MIN_CLOSE_HEIGHT_RATIO,
    MODEL_PATH,
)


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
