# Configuration settings for video inspection system
# =================================================
# Architecture: YOLO Detection → Frame Gating → Vision Analysis → Rule Engine
import os
from pathlib import Path

# ============================================================
# Model Backend Selection
# ============================================================

# Toggle between local models (LLaVA via Ollama) and API models (Claude)
# Set USE_LOCAL_MODELS=true in environment to use local Ollama models
USE_LOCAL_MODELS = os.getenv("USE_LOCAL_MODELS", "false").lower() == "true"

# ============================================================
# Ollama Configuration (Local Models)
# ============================================================

OLLAMA_IP = os.getenv("OLLAMA_IP", "localhost")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_API_URL = f"http://{OLLAMA_IP}:{OLLAMA_PORT}/api/generate"
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava:13b")

# ============================================================
# Claude API Configuration
# ============================================================

# Claude API key - loaded from environment variable
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Claude model to use for vision analysis
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# ============================================================
# Incident Categories Configuration
# ============================================================

INCIDENT_CATEGORIES = [
    "Unauthorised Person Detected",
    "Armed Attack Detected",
    "Crowd Detected",
    "Violence Detected",
    "Accident Detected",
    "Fire Detected",
    "Electric Hazard Detected",
    "Maintenance Issue Detected",
    "No Incident Detected",
]

# ============================================================
# Dispatch Actions Configuration
# ============================================================

DISPATCH_ACTIONS = {
    "DRONE": ("Dispatch Drone", "For surveillance and monitoring"),
    "SMALL_TEAM": ("Dispatch Small Team", "Small team of 2-3 security staff"),
    "ARMED_TEAM": ("Dispatch Armed Team", "Large team prepared for armed situations"),
    "AMBULANCE": ("Dispatch Ambulance Service", "Alert medical services"),
    "FIRE_SERVICE": ("Dispatch Fire Service", "Alert fire department"),
    "MAINTENANCE": (
        "Dispatch Maintenance Team",
        "For electric hazards, chemical leakages, and maintenance issues",
    ),
}

# ============================================================
# Scenario to Action Mapping Configuration
# ============================================================

SCENARIO_ACTION_MAPPING = {
    "Unauthorised Person Detected": ["DRONE", "SMALL_TEAM"],
    "Armed Attack Detected": ["DRONE", "ARMED_TEAM"],
    "Crowd Detected": ["DRONE", "SMALL_TEAM"],
    "Violence Detected": ["DRONE", "SMALL_TEAM"],
    "Accident Detected": ["DRONE", "SMALL_TEAM", "AMBULANCE"],
    "Fire Detected": ["DRONE", "SMALL_TEAM", "FIRE_SERVICE"],
    "Electric Hazard Detected": ["DRONE", "MAINTENANCE"],
    "Maintenance Issue Detected": ["MAINTENANCE"],
    "No Incident Detected": [],
}


# ============================================================
# Helper Functions to Build Prompt Sections
# ============================================================


def _build_incident_categories_section():
    """Build the incident categories section for the prompt."""
    lines = ["- incident_category: One of"]
    for category in INCIDENT_CATEGORIES:
        lines.append(f'  - "{category}"')
    return "\n".join(lines)


def _build_dispatch_actions_section():
    """Build the dispatch actions section for the prompt."""
    lines = [
        "- recommended_action:",
        "  - Array of actions to dispatch based on incident_category",
        "  - Choose from the following based on scenario:",
    ]
    for key, (action_name, description) in DISPATCH_ACTIONS.items():
        lines.append(f'    - "{action_name}" - {description}')
    return "\n".join(lines)


def _build_scenario_mapping_section():
    """Build the scenario to action mapping section for the prompt."""
    lines = ["  ", "  SCENARIO TO ACTION MAPPING:"]
    for scenario, action_keys in SCENARIO_ACTION_MAPPING.items():
        action_names = [DISPATCH_ACTIONS[key][0] for key in action_keys]
        lines.append(f"  - {scenario} → {action_names}")
    return "\n".join(lines)


# ============================================================
# Shared Analysis Prompt (Used by both Local and API analyzers)
# ============================================================

# Base prompt template
_ANALYSIS_PROMPT_TEMPLATE = """
You are a professional security officer monitoring surveillance footage from a RESTRICTED AREA.
Unauthorized access is not permitted.

TASK:
Analyze the provided video and generate a structured security incident report.
At this place and time no human activity should happen as this is a restricted area.

IMPORTANT RULES:
- Describe ONLY what is clearly visible in the video
- Do NOT guess intent, identity, or authorization
- Do NOT include assumptions unless explicitly visible (badge, escort, signage)
- Focus on people, their actions, and interactions with doors, locks, gates, or access points
- Be factual, concise, and security-oriented
- If no suspicious activity is visible, explicitly state that no incident is detected

OUTPUT REQUIREMENTS (STRICT):
- Output MUST be valid JSON
- Do NOT include markdown, explanations, or extra text
- Do NOT add additional fields
- Do NOT include comments

JSON SCHEMA (FOLLOW EXACTLY):

{{
  "incident_category": "",
  "observed_activity": [],
  "risk_assessment": "",
  "recommended_action": [],
  "next_action": []
}}

FIELD GUIDELINES:
{incident_categories}

- observed_activity:
  - Array of short, factual strings
  - Describe visible actions only

- risk_assessment:
  - One of: "SAFE", "POTENTIALLY UNSAFE", "UNSAFE"
  
{dispatch_actions}
{scenario_mapping}

- next_action:
  - Array of clear, actionable steps for security personnel
  - No speculation
"""

# Build the final prompt dynamically from configuration
ANALYSIS_SYSTEM_PROMPT = _ANALYSIS_PROMPT_TEMPLATE.format(
    incident_categories=_build_incident_categories_section(),
    dispatch_actions=_build_dispatch_actions_section(),
    scenario_mapping=_build_scenario_mapping_section(),
)

# ============================================================
# Model Configuration
# ============================================================

# YOLOv8 model for object detection
YOLO_MODEL_PATH = "best.pt"

# Minimum confidence threshold for detections
DETECTION_CONFIDENCE_THRESHOLD = 0.4

# ============================================================
# COCO Class IDs for Detection
# ============================================================

# Person class ID
PERSON_CLASS_ID = 3

# Classes to always detect (subset of COCO 80 classes)
# Format: {class_id: (class_name, is_dangerous)}
DETECTION_CLASSES = {
    0: ("Fire", True),
    1: ("Gun", True),
    2: ("NonViolence", False),
    3: ("Person", False),
    4: ("Smoke", False),
    7: ("Violence", True),
    8: ("knife", True),
}

# IDs of dangerous objects (triggers HIGH priority)
DANGEROUS_CLASS_IDS = {0, 1, 7, 8}

# ============================================================
# Frame Extraction Configuration
# ============================================================

# Frames per second for extraction
DEFAULT_FPS = 1

# Maximum frames to extract from video
MAX_FRAMES = 30

# Maximum frames to send to Vision Model (cost control)
MAX_FRAMES_FOR_VISION = 8

# Temporary directory for extracted frames
TEMP_FRAMES_DIR = Path("./temp_frames")

# ============================================================
# Safety Classification Rules
# ============================================================

# Safety levels
SAFETY_SAFE = "SAFE"
SAFETY_UNSAFE = "UNSAFE"
SAFETY_REVIEW = "REVIEW"

# Rule: If Vision Model reports HIGH risk, mark as UNSAFE
# Rule: If dangerous object detected by YOLO, mark as UNSAFE
# Rule: If no person detected, mark as SAFE
