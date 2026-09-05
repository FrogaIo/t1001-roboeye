from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from config import (
    DEPTH_INTERVAL_SECONDS,
    DEPTH_MAX_AGE_SECONDS,
    DEPTH_MODEL_ID,
)


logger = logging.getLogger("roboeye.depth")


@dataclass(frozen=True)
class DepthObservation:
    """Latest relative depth map. Values are unitless: higher means closer.

    They must never be interpreted as metres without calibration."""

    depth: np.ndarray
    created_at: float
    inference_ms: float

    def age_seconds(self, now: float) -> float:
        return now - self.created_at


def observation_is_fresh(observation: DepthObservation | None, now: float) -> bool:
    """A depth map older than DEPTH_MAX_AGE_SECONDS is never fresh evidence."""
    return (
        observation is not None
        and observation.age_seconds(now) <= DEPTH_MAX_AGE_SECONDS
    )


class DepthDetector:
    """Depth Anything V2 Small. The model is loaded once and kept on device."""

    def __init__(self) -> None:
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID)
            .to(self.device)
            .eval()
        )

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        height, width = frame_bgr.shape[:2]
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            prediction = self.model(**inputs).predicted_depth
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            )[0, 0]
        return prediction.float().cpu().numpy()


class DepthTracker:
    """Runs depth inference at most once per interval and reuses the last map.

    Failures never raise: the system falls back to the YOLO-only mode."""

    def __init__(self) -> None:
        self._detector: DepthDetector | None = None
        self._disabled = False
        self._latest: DepthObservation | None = None
        self._last_started_at = 0.0

    def observe(
        self, frame_bgr: np.ndarray, now: float | None = None
    ) -> DepthObservation | None:
        now = time.monotonic() if now is None else now
        if self._disabled:
            return self._latest
        if self._detector is None:
            try:
                self._detector = DepthDetector()
            except Exception:
                logger.exception("[depth] model load failed; continuing YOLO-only")
                self._disabled = True
                return self._latest
        if (
            self._latest is not None
            and now - self._last_started_at < DEPTH_INTERVAL_SECONDS
        ):
            return self._latest

        self._last_started_at = now
        started = time.perf_counter()
        try:
            depth = self._detector.infer(frame_bgr)
        except Exception:
            logger.exception("[depth] inference failed; keeping previous map")
            return self._latest
        observation = DepthObservation(
            depth=depth,
            created_at=now,
            inference_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        self._latest = observation
        return observation
