# Configuration settings for video inspection system
import os
from pathlib import Path

# ============================================================
# API Configuration
# ============================================================

# Claude API key - loaded from environment variable
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Claude model to use for summarization
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# Claude prompt
CLAUDE_PROMPT = """You are a security footage analysis assistant.
Your task is to convert structured detection facts into a brief natural language summary.

RULES:
1. Output EXACTLY 2-3 sentences, no more
2. Only state what is provided in the facts - do NOT speculate or add details
3. Use simple, clear language
4. Focus on: who was detected, what they did, and for how long
5. If actions are detected, mention the most prominent one
6. Do NOT make assumptions about intent or identity"""

# ============================================================
# Model Configuration
# ============================================================

# YOLOv8 model for person detection
YOLO_MODEL_PATH = "yolov8n.pt"

# VideoMAE model for action recognition (Hugging Face)
VIDEOMAE_MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"

# Person class ID in COCO dataset (used by YOLO)
PERSON_CLASS_ID = 0

# Minimum confidence threshold for person detection
PERSON_CONFIDENCE_THRESHOLD = 0.5

# ============================================================
# Frame Extraction Configuration
# ============================================================

# Default frames per second for extraction
DEFAULT_FPS = 1

# Temporary directory for extracted frames
TEMP_FRAMES_DIR = Path("./temp_frames")

# Number of frames for action recognition (VideoMAE expects 16 frames)
ACTION_RECOGNITION_NUM_FRAMES = 16

# ============================================================
# Action Recognition Configuration
# ============================================================

# Top-k actions to consider from model predictions
TOP_K_ACTIONS = 3

# Minimum confidence for action to be considered valid
ACTION_CONFIDENCE_THRESHOLD = 0.2
