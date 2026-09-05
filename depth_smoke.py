"""Smoke test for Depth Anything V2 Small on Apple MPS.

Verifies that the model loads, runs on the selected device and produces a
plausible dense depth map. Also answers the mandatory experiment from
DEPTH_PLAN.md section 2: whether higher predicted values mean closer or
farther surfaces. The sign must come from this experiment, not assumption.

Usage:
    source .venv-depth/bin/activate
    python depth_smoke.py [--near x1,y1,x2,y2] [--far x1,y1,x2,y2]

Region coordinates are fractions of width/height in the sample image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
INPUT = Path("sample.jpg")
OUTPUT = Path("depth-preview.jpg")


def parse_region(value: str) -> tuple[float, float, float, float]:
    parts = [float(part) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("region must be x1,y1,x2,y2 fractions")
    return tuple(parts)  # type: ignore[return-value]


def region_median(depth: np.ndarray, region: tuple[float, float, float, float]) -> float:
    height, width = depth.shape
    x1, y1, x2, y2 = region
    crop = depth[
        int(y1 * height) : int(y2 * height),
        int(x1 * width) : int(x2 * width),
    ]
    return float(np.median(crop))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--near",
        type=parse_region,
        default=(0.60, 0.40, 0.98, 0.85),
        help="region of a clearly close object (default: close table in sample)",
    )
    parser.add_argument(
        "--far",
        type=parse_region,
        default=(0.28, 0.44, 0.45, 0.52),
        help="region of clearly far floor (default: distant floor in sample)",
    )
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device).eval()

    image = Image.open(INPUT).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    # Warm-up run excluded from timing.
    with torch.inference_mode():
        model(**inputs)

    timings = []
    for _ in range(3):
        started = torch.mps.Event(enable_timing=True) if device == "mps" else None
        ended = torch.mps.Event(enable_timing=True) if device == "mps" else None
        if started is not None and ended is not None:
            started.record()
        with torch.inference_mode():
            prediction = model(**inputs).predicted_depth
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=(image.height, image.width),
                mode="bicubic",
                align_corners=False,
            )[0, 0]
        if started is not None and ended is not None:
            ended.record()
            torch.mps.synchronize()
            timings.append(started.elapsed_time(ended))

    depth = prediction.float().cpu().numpy()
    low, high = np.percentile(depth, (2, 98))
    normalized = np.clip((depth - low) / max(high - low, 1e-6), 0, 1)
    preview = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    cv2.imwrite(str(OUTPUT), preview)

    print(f"device={device} shape={depth.shape} range=({depth.min():.3f}, {depth.max():.3f})")
    if timings:
        print(f"inference ms per run: {[round(t, 1) for t in timings]}")

    near_median = region_median(depth, args.near)
    far_median = region_median(depth, args.far)
    print(f"near region median={near_median:.3f} far region median={far_median:.3f}")
    if near_median > far_median:
        print("SIGN: higher value = CLOSER (depth-like)")
    else:
        print("SIGN: higher value = FARTHER (inverse-depth-like)")
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
