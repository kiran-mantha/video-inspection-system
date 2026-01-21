# Configuration settings for video inspection system
# =================================================
# Architecture: YOLO Detection → Frame Gating → BLIP API → Rule Engine
import os
from pathlib import Path

# ============================================================
# BLIP API Configuration (Remote Docker Server)
# ============================================================

# BLIP API server URL (running in Docker)
# Format: http://<server-ip>:<port>
# Examples:
#   - http://localhost:5000 (if running locally)
#   - http://192.168.1.100:5000 (remote server)
#   - http://blip-server:5000 (Docker network)
BLIP_API_URL = os.getenv("BLIP_API_URL", "http://100.65.72.122:5007")

# API endpoints
BLIP_ANALYZE_ENDPOINT = f"{BLIP_API_URL}/analyze"
BLIP_HEALTH_ENDPOINT = f"{BLIP_API_URL}/health"

# Request timeout in seconds
BLIP_API_TIMEOUT = 30

# ============================================================
# Risk Assessment Keywords
# ============================================================

# Keywords that indicate potential danger (for risk assessment)
DANGER_KEYWORDS = [
    "gun",
    "pistol",
    "rifle",
    "firearm",
    "weapon",
    "shooting",
    "knife",
    "blade",
    "sword",
    "stabbing",
    "fighting",
    "attack",
    "hitting",
    "punching",
    "kicking",
    "blood",
    "injured",
    "violence",
    "aggressive",
    "masked",
    "robbery",
    "stealing",
    "threat",
]

MEDIUM_RISK_KEYWORDS = [
    "running",
    "chasing",
    "arguing",
    "yelling",
    "suspicious",
    "hiding",
    "crawling",
    "climbing",
    "breaking",
]

# ============================================================
# YOLO Model Configuration
# ============================================================

# YOLOv8 model for object detection
YOLO_MODEL_PATH = "yolov8n.pt"

# Minimum confidence threshold for detections
DETECTION_CONFIDENCE_THRESHOLD = 0.4

# ============================================================
# COCO Class IDs for Detection
# ============================================================

# Person class ID
PERSON_CLASS_ID = 0

# Classes to always detect (subset of COCO 80 classes)
# Format: {class_id: (class_name, is_dangerous)}
DETECTION_CLASSES = {
    0: ("person", False),
    24: ("backpack", False),
    25: ("umbrella", False),
    26: ("handbag", False),
    27: ("tie", False),
    28: ("suitcase", False),
    39: ("bottle", False),
    43: ("knife", True),  # DANGEROUS
    76: ("scissors", True),  # DANGEROUS
}

# IDs of dangerous objects (triggers HIGH priority)
DANGEROUS_CLASS_IDS = {43, 76}  # knife, scissors

# ============================================================
# Frame Extraction Configuration
# ============================================================

# Frames per second for extraction
DEFAULT_FPS = 1

# Maximum frames to extract from video
MAX_FRAMES = 30

# Maximum frames to send to vision model (cost/performance control)
MAX_FRAMES_FOR_VISION = 3

# Temporary directory for extracted frames
TEMP_FRAMES_DIR = Path("./temp_frames")

# ============================================================
# Safety Classification Rules
# ============================================================

# Safety levels
SAFETY_SAFE = "SAFE"
SAFETY_UNSAFE = "UNSAFE"
SAFETY_REVIEW = "REVIEW"
