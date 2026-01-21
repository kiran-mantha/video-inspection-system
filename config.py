# Configuration settings for video inspection system
# =================================================
# Architecture: YOLO Detection → Frame Gating → Claude Vision → Rule Engine
import os
from pathlib import Path

# ============================================================
# API Configuration
# ============================================================

# Claude API key - loaded from environment variable
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Claude model to use for vision analysis
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# Claude Vision prompt for frame analysis
CLAUDE_VISION_PROMPT = """You are a security footage analyst. Analyze the provided frame(s) from surveillance footage.

TASK:
1. Describe what you observe in 1-2 concise sentences
2. Focus on: people present, their actions, and any objects they are holding or interacting with
3. Note any potentially dangerous items (weapons, suspicious objects)

RULES:
- Only describe what is VISIBLE in the image(s)
- Do NOT speculate about intent or identity
- Do NOT make assumptions about what happened before or after
- Be factual and objective
- If you see a weapon (gun, knife, etc.), explicitly mention it

OUTPUT FORMAT:
Observation: [1-2 sentences describing the scene]
Objects: [list key objects detected, especially any weapons]
Risk_Level: [LOW / MEDIUM / HIGH]"""

# ============================================================
# Model Configuration
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
    # Note: COCO does not have "gun" - Claude Vision will identify firearms
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

# Maximum frames to send to Claude Vision (cost control)
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

# Rule: If Claude reports HIGH risk, mark as UNSAFE
# Rule: If dangerous object detected by YOLO, mark as UNSAFE
# Rule: If no person detected, mark as SAFE
