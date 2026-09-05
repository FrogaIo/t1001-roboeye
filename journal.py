from __future__ import annotations

import json
import threading
from collections import deque
from datetime import datetime
from pathlib import Path


class EventJournal:
    """Append-only state journal and incident snapshots stored on the Mac."""

    def __init__(self, root: Path, keep_recent: int = 50) -> None:
        self.log_dir = root / "logs"
        self.incident_dir = root / "incidents"
        self.log_path = self.log_dir / "events.jsonl"
        self.log_dir.mkdir(exist_ok=True)
        self.incident_dir.mkdir(exist_ok=True)
        self._events: deque[dict[str, object]] = deque(maxlen=keep_recent)
        self._lock = threading.Lock()
        self._load_recent()

    def _load_recent(self) -> None:
        if not self.log_path.exists():
            return
        try:
            lines = self.log_path.read_text(encoding="utf-8").splitlines()
            for line in reversed(lines[-self._events.maxlen :]):
                self._events.append(json.loads(line))
        except (OSError, json.JSONDecodeError):
            # A broken old log must never prevent the vision system from starting.
            self._events.clear()

    def record_transition(
        self,
        *,
        previous: str,
        current: str,
        reason: str | None,
        route: str,
        detections: list[dict[str, object]],
        annotated_jpeg: bytes,
    ) -> dict[str, object]:
        now = datetime.now().astimezone()
        incident_name: str | None = None
        if current == "STOP":
            incident_name = f"incident-{now.strftime('%Y%m%d-%H%M%S-%f')}.jpg"
            (self.incident_dir / incident_name).write_bytes(annotated_jpeg)

        event: dict[str, object] = {
            "timestamp": now.isoformat(timespec="seconds"),
            "time": now.strftime("%H:%M:%S"),
            "from": previous,
            "to": current,
            "reason": reason or "zone clear",
            "route": route,
            "objects": [
                {
                    "label": item["label"],
                    "confidence": item["confidence"],
                    "lane": item["lane"],
                }
                for item in detections
            ],
            "incident": incident_name,
        }

        with self._lock:
            self._events.appendleft(event)
            with self.log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    def recent(self) -> list[dict[str, object]]:
        with self._lock:
            return list(self._events)
