from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from config import (
    CLEAR_FRAMES_REQUIRED,
    DARK_FRAME_MEAN_THRESHOLD,
    DEFAULT_CAMERA_PITCH_DEGREES,
    DISPLAY_NAMES,
)
from detector import (
    DecisionFilter,
    ObjectDetector,
    choose_route,
    draw_result,
    lane_occupancy,
    perspective_for_pitch,
)
from depth_detector import DepthTracker
from journal import EventJournal


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("roboeye")

app = FastAPI(title="RoboEye")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

latest_processed_frame: bytes | None = None
latest_status: dict[str, object] | None = None
latest_processed_at = 0.0

detector = ObjectDetector()
depth_tracker = DepthTracker()
journal = EventJournal(ROOT)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/monitor")
async def monitor() -> FileResponse:
    return FileResponse(STATIC_DIR / "monitor.html")


@app.get("/latest.jpg")
async def latest_frame() -> Response:
    if latest_processed_frame is None or time.monotonic() - latest_processed_at > 2.0:
        return Response(status_code=204, headers={"Cache-Control": "no-store"})
    return Response(
        content=latest_processed_frame,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/api/status")
async def current_status() -> JSONResponse:
    connected = latest_status is not None and time.monotonic() - latest_processed_at <= 2.0
    return JSONResponse(
        {
            "connected": connected,
            "status": latest_status if connected else None,
            "events": journal.recent(),
        },
        headers={"Cache-Control": "no-store"},
    )


@app.get("/incidents/{filename}")
async def incident(filename: str) -> FileResponse:
    if Path(filename).name != filename or not filename.endswith(".jpg"):
        raise HTTPException(status_code=404)
    path = journal.incident_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="image/jpeg")


def process_frame(
    jpeg: bytes,
    decision_filter: DecisionFilter,
    pitch_degrees: float | None = None,
) -> tuple[bytes, dict[str, object]] | None:
    started_at = time.perf_counter()
    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None

    height, width = frame.shape[:2]
    if width > 720:
        scale = 720 / width
        frame = cv2.resize(frame, (720, int(height * scale)))

    perspective = perspective_for_pitch(pitch_degrees)
    too_dark = float(frame.mean()) < DARK_FRAME_MEAN_THRESHOLD
    detections = [] if too_dark else detector.detect(frame, perspective)
    depth = depth_tracker.observe(frame)
    lanes = lane_occupancy(detections)
    route = "BLOCKED" if too_dark else choose_route(lanes)

    dangerous = [item for item in detections if item.dangerous]
    danger_reason = "camera_dark" if too_dark else (dangerous[0].label if dangerous else None)
    decision = decision_filter.update(danger_reason)
    processed = draw_result(frame, detections, decision, lanes, route, perspective)

    ok, encoded = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not ok:
        return None

    height, width = frame.shape[:2]
    payload: dict[str, object] = {
        "state": decision.state,
        "reason": decision.reason,
        "reason_display": DISPLAY_NAMES.get(decision.reason, "ZONE CLEAR"),
        "route": route,
        "lanes": lanes,
        "inference_ms": round((time.perf_counter() - started_at) * 1000),
        "pitch_degrees": round(perspective.pitch_degrees, 1),
        "zone": [0.0, perspective.zone_top, 1.0, 1.0],
        "corridor": {
            "top_left": [perspective.top_left, perspective.zone_top],
            "top_right": [perspective.top_right, perspective.zone_top],
            "bottom_right": [perspective.bottom_right, 1.0],
            "bottom_left": [perspective.bottom_left, 1.0],
        },
        "transition_from": decision.changed_from,
        "depth": {
            "available": depth is not None,
            "age_ms": (
                int(depth.age_seconds(time.monotonic()) * 1000) if depth else None
            ),
            "inference_ms": depth.inference_ms if depth else None,
        },
        "detections": [
            {
                "label": item.label,
                "display_name": DISPLAY_NAMES[item.label],
                "confidence": round(item.confidence, 3),
                "box": [
                    item.box[0] / width,
                    item.box[1] / height,
                    item.box[2] / width,
                    item.box[3] / height,
                ],
                "lane": item.lane,
                "close": item.close,
                "dangerous": item.dangerous,
            }
            for item in detections
        ],
    }
    return encoded.tobytes(), payload


@app.websocket("/ws")
async def camera_round_trip(websocket: WebSocket) -> None:
    global latest_processed_at, latest_processed_frame, latest_status

    await websocket.accept()
    decision_filter = DecisionFilter(clear_frames_required=CLEAR_FRAMES_REQUIRED)
    pitch_degrees = DEFAULT_CAMERA_PITCH_DEGREES
    client = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"
    logger.info("[camera] connected %s", client)

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(code=message.get("code", 1000))
            if message.get("text") is not None:
                try:
                    metadata = json.loads(message["text"])
                    if metadata.get("type") == "orientation":
                        pitch_degrees = float(metadata.get("pitch_degrees"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    logger.warning("[camera] invalid orientation metadata from %s", client)
                continue

            jpeg = message.get("bytes")
            if jpeg is None:
                continue
            result = await asyncio.to_thread(
                process_frame, jpeg, decision_filter, pitch_degrees
            )
            if result is None:
                logger.warning("[camera] invalid or empty frame from %s", client)
                await websocket.send_json({"error": "invalid frame"})
                continue

            processed, payload = result
            latest_processed_frame = processed
            latest_processed_at = time.monotonic()
            latest_status = payload

            detections = payload["detections"]
            summary = ", ".join(
                f"{item['label']} {item['confidence']:.2f} lane={item['lane']}"
                for item in detections
            ) or "clear"
            logger.info(
                "[detect] %s | state=%s route=%s pitch=%.1f° inference=%sms",
                summary,
                payload["state"],
                payload["route"],
                payload["pitch_degrees"],
                payload["inference_ms"],
            )

            if payload["transition_from"] is not None:
                event = journal.record_transition(
                    previous=str(payload["transition_from"]),
                    current=str(payload["state"]),
                    reason=payload["reason"] if isinstance(payload["reason"], str) else None,
                    route=str(payload["route"]),
                    detections=detections,
                    annotated_jpeg=processed,
                )
                logger.info(
                    "[state] %s -> %s reason=%s route=%s incident=%s",
                    event["from"],
                    event["to"],
                    event["reason"],
                    event["route"],
                    event["incident"] or "none",
                )

            await websocket.send_json(payload)
    except WebSocketDisconnect:
        logger.info("[camera] disconnected %s", client)
    except Exception:
        logger.exception("[error] camera processing failed for %s", client)
        await websocket.close(code=1011)
