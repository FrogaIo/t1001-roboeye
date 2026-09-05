from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "yolov8s-worldv2.pt"

# Several text prompts may represent one logical class. This improves the
# zero-shot detector on unusual branded and colourful boxes without training.
CLASS_PROMPTS = {
    "person": ("person",),
    "chair": ("chair",),
    "table": ("table", "dining table"),
    "box": (
        "cardboard box",
        "colorful cardboard box",
        "cube shaped box",
        "cardboard display cube",
        "package box",
        "storage box",
    ),
    "pallet": ("wooden pallet",),
    "forklift": ("forklift",),
    "cart": ("warehouse cart", "trolley"),
    "bin": ("trash bin", "storage bin"),
}

DISPLAY_NAMES = {
    "person": "PERSON",
    "chair": "CHAIR",
    "table": "TABLE",
    "box": "BOX",
    "pallet": "PALLET",
    "forklift": "FORKLIFT",
    "cart": "CART",
    "bin": "BIN",
    "camera_dark": "CAMERA DARK",
}

CONFIDENCE_THRESHOLD = 0.18
IOU_THRESHOLD = 0.45
INFERENCE_SIZE = 640
MAX_DETECTIONS = 24

# Monocular depth. The model outputs RELATIVE depth: values are unitless and
# higher value means a CLOSER surface (verified experimentally in
# depth_smoke.py). Never interpret them as metres.
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_INTERVAL_SECONDS = 0.5

MIN_CLOSE_HEIGHT_RATIO = 0.18
# Perspective is derived from the phone accelerometer. The fallback occupies
# roughly the lower third of the image when motion access is unavailable.
DEFAULT_CAMERA_PITCH_DEGREES = 20.0
MIN_CAMERA_PITCH_DEGREES = 0.0
MAX_CAMERA_PITCH_DEGREES = 75.0
CONTROL_ZONE_TOP_UPRIGHT = 0.72
CONTROL_ZONE_TOP_DOWN = 0.64
CORRIDOR_TOP_HALF_WIDTH_UPRIGHT = 0.10
CORRIDOR_TOP_HALF_WIDTH_DOWN = 0.26
CORRIDOR_BOTTOM_LEFT = 0.04
CORRIDOR_BOTTOM_RIGHT = 0.96
CLEAR_FRAMES_REQUIRED = 3
DARK_FRAME_MEAN_THRESHOLD = 10.0
