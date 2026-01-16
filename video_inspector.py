# Video Inspection System
# ========================
# Main module for video surveillance analysis using:
# - YOLOv8n for person detection
# - VideoMAE Transformer for action recognition
# - Claude API for natural language summary generation

import os
from typing import List, Dict
from pathlib import Path

import cv2
import numpy as np

from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    YOLO_MODEL_PATH,
    PERSON_CLASS_ID,
    PERSON_CONFIDENCE_THRESHOLD,
    DEFAULT_FPS,
    TOP_K_ACTIONS,
    ACTION_CONFIDENCE_THRESHOLD,
    CLAUDE_PROMPT,
)
from utils import (
    ensure_temp_dir,
    cleanup_frames,
    calculate_timestamp,
    format_duration,
)
from action_recognizer import recognize_actions


# ============================================================
# Function 1: Extract Frames from Video
# ============================================================


def extract_frames(video_path: str, fps: int = DEFAULT_FPS) -> List[str]:
    """
    Extract frames from a video file at the specified FPS rate.

    Uses OpenCV to read the video and extract frames uniformly
    based on the target FPS. Frames are saved as PNG images in
    a temporary directory.

    Args:
        video_path: Absolute path to the video file
        fps: Target frames per second to extract (default: 1)

    Returns:
        List of absolute paths to extracted frame images

    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video cannot be opened or has no frames
    """
    # Validate input
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")

    # Calculate frame interval based on target FPS
    # If video is 30fps and we want 1fps, we take every 30th frame
    frame_interval = max(1, int(video_fps / fps))

    # Prepare output directory
    temp_dir = ensure_temp_dir()
    video_name = Path(video_path).stem

    frame_paths = []
    frame_index = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract frame at the interval
        if frame_index % frame_interval == 0:
            # Save frame as PNG
            frame_filename = f"{video_name}_frame_{extracted_count:05d}.png"
            frame_path = str(temp_dir / frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            extracted_count += 1

        frame_index += 1

    cap.release()

    return frame_paths


# ============================================================
# Function 2: Detect People in Frames
# ============================================================


def detect_people(frame_paths: List[str]) -> List[Dict]:
    """
    Run YOLOv8n person detection on extracted frames.

    Loads the YOLOv8n model and processes each frame to detect
    people. Only detections with confidence above the threshold
    are returned.

    Args:
        frame_paths: List of absolute paths to frame images

    Returns:
        List of detection dictionaries, each containing:
        - frame_index: Index of the frame (0-based)
        - timestamp: Estimated timestamp in seconds
        - bounding_boxes: List of [x1, y1, x2, y2] coordinates
        - confidences: List of detection confidence scores
    """
    from ultralytics import YOLO

    # Load YOLOv8n model
    model = YOLO(YOLO_MODEL_PATH)

    detections = []

    for idx, frame_path in enumerate(frame_paths):
        # Run inference on the frame
        results = model(frame_path, verbose=False)

        frame_boxes = []
        frame_confidences = []

        # Process results for this frame
        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Check if detection is a person (class 0)
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if (
                    cls_id == PERSON_CLASS_ID
                    and confidence >= PERSON_CONFIDENCE_THRESHOLD
                ):
                    # Get bounding box coordinates
                    xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    frame_boxes.append([round(c, 2) for c in xyxy])
                    frame_confidences.append(round(confidence, 3))

        # Only add frame to detections if people were found
        if frame_boxes:
            detections.append(
                {
                    "frame_index": idx,
                    "timestamp": calculate_timestamp(idx, DEFAULT_FPS),
                    "bounding_boxes": frame_boxes,
                    "confidences": frame_confidences,
                }
            )

    return detections


# ============================================================
# Function 3: Infer Actions from Detections
# ============================================================


def infer_actions(detections: List[Dict], frame_paths: List[str] = None) -> Dict:
    """
    Infer actions using VideoMAE transformer model.

    This function uses a pre-trained action recognition model
    to identify what actions people are performing in the video.
    If frame_paths are provided, it uses VideoMAE for sophisticated
    action recognition. Otherwise, it falls back to basic analysis.

    Args:
        detections: List of detection dictionaries from detect_people()
        frame_paths: Optional list of frame paths for action recognition

    Returns:
        Dictionary containing:
        - person_detected: Boolean indicating if any person was found
        - total_detections: Number of frames with people
        - first_appearance: Timestamp of first person detection
        - last_appearance: Timestamp of last person detection
        - duration_visible: How long person was visible
        - actions: List of recognized actions with confidence
        - movement_analysis: Basic movement analysis from bounding boxes
    """
    if not detections:
        return {
            "person_detected": False,
            "total_detections": 0,
            "first_appearance": None,
            "last_appearance": None,
            "duration_visible": 0,
            "actions": [],
            "movement_analysis": None,
        }

    # Basic timing analysis from detections
    first_appearance = detections[0]["timestamp"]
    last_appearance = detections[-1]["timestamp"]
    duration_visible = last_appearance - first_appearance

    # Movement analysis from bounding boxes
    movement = _analyze_movement(detections)

    # Action recognition using VideoMAE
    recognized_actions = []
    if frame_paths:
        # Use VideoMAE for sophisticated action recognition
        recognized_actions = recognize_actions(
            frame_paths,
            top_k=TOP_K_ACTIONS,
            confidence_threshold=ACTION_CONFIDENCE_THRESHOLD,
        )

    return {
        "person_detected": True,
        "total_detections": len(detections),
        "first_appearance": first_appearance,
        "last_appearance": last_appearance,
        "duration_visible": round(duration_visible, 2),
        "actions": recognized_actions,
        "movement_analysis": movement,
    }


def _analyze_movement(detections: List[Dict]) -> Dict:
    """
    Analyze movement patterns from bounding box positions.

    This provides supplementary movement information based on
    how bounding boxes change position across frames.

    Args:
        detections: List of detection dictionaries

    Returns:
        Dictionary with movement analysis
    """
    if len(detections) < 2:
        return {"pattern": "stationary", "displacement": 0}

    # Track center points of first detected person across frames
    centers = []
    for det in detections:
        if det["bounding_boxes"]:
            box = det["bounding_boxes"][0]  # First person
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            centers.append((center_x, center_y))

    if len(centers) < 2:
        return {"pattern": "stationary", "displacement": 0}

    # Calculate total displacement
    total_displacement = 0
    for i in range(1, len(centers)):
        dx = centers[i][0] - centers[i - 1][0]
        dy = centers[i][1] - centers[i - 1][1]
        total_displacement += np.sqrt(dx**2 + dy**2)

    # Classify movement pattern
    avg_displacement = total_displacement / (len(centers) - 1)

    if avg_displacement < 20:
        pattern = "stationary"
    elif avg_displacement < 50:
        pattern = "minimal movement"
    elif avg_displacement < 100:
        pattern = "walking"
    else:
        pattern = "significant movement"

    return {
        "pattern": pattern,
        "displacement": round(total_displacement, 2),
    }


# ============================================================
# Function 4: Build Structured Summary
# ============================================================


def build_summary(action_data: Dict) -> Dict:
    """
    Convert action inference data into a compact factual summary.

    This function transforms the detailed action analysis into
    a simplified JSON structure that can be sent to Claude for
    natural language generation. It contains ONLY facts, no
    interpretations.

    Args:
        action_data: Dictionary from infer_actions()

    Returns:
        Compact summary dictionary with:
        - person_present: Boolean
        - visibility_duration: Human-readable duration string
        - detected_actions: List of action labels
        - movement: Movement pattern description
    """
    if not action_data.get("person_detected", False):
        return {
            "person_present": False,
            "visibility_duration": None,
            "detected_actions": [],
            "movement": None,
        }

    # Extract action labels (without confidence scores for Claude)
    action_labels = [action["action"] for action in action_data.get("actions", [])]

    # Format duration for readability
    duration = action_data.get("duration_visible", 0)
    duration_str = format_duration(duration) if duration > 0 else "briefly"

    # Get movement pattern
    movement = action_data.get("movement_analysis", {})
    movement_pattern = movement.get("pattern", "unknown") if movement else "unknown"

    return {
        "person_present": True,
        "visibility_duration": duration_str,
        "detected_actions": action_labels,
        "movement": movement_pattern,
    }


# ============================================================
# Function 5: Summarize with Claude API
# ============================================================


def summarize_with_claude(summary: Dict) -> str:
    """
    Generate a natural language summary using Claude API.

    Claude receives ONLY the structured JSON facts and converts
    them into 2-3 sentences. Claude does NOT receive any images
    or video data, and is instructed not to speculate.

    Args:
        summary: Structured summary dictionary from build_summary()

    Returns:
        Human-readable verdict string (2-3 sentences max)

    Raises:
        ValueError: If ANTHROPIC_API_KEY is not set
    """
    import json
    from anthropic import Anthropic

    # Check for API key
    api_key = ANTHROPIC_API_KEY
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Please set it to use Claude for summarization."
        )

    # If no person detected, return safe message directly
    if not summary.get("person_present", False):
        return "The footage is safe."

    # Prepare the prompt for Claude
    system_prompt = CLAUDE_PROMPT

    # Format facts as JSON for Claude
    facts_json = json.dumps(summary, indent=2)

    user_prompt = f"""Convert these detection facts into a 2-3 sentence summary:

{facts_json}

Remember: Only state what's in the facts. Do not speculate."""

    # Call Claude API
    client = Anthropic(api_key=api_key)

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=150,  # Keep response short
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    # Extract text from response
    return response.content[0].text.strip()


# ============================================================
# Function 6: Main Orchestration Function
# ============================================================


def inspect_video(video_path: str) -> str:
    """
    Main function to inspect a video for people and their actions.

    This orchestrates all the detection and analysis steps:
    1. Extract frames from video
    2. Detect people using YOLOv8n
    3. Infer actions using VideoMAE transformer
    4. Build structured summary
    5. Generate natural language verdict using Claude

    Args:
        video_path: Absolute path to the video file to inspect

    Returns:
        Human-readable inspection verdict (2-3 sentences)
        If no people detected: "The footage is safe."
    """
    frame_paths = []

    try:
        # Step 1: Extract frames from video
        print(f"Extracting frames from: {video_path}")
        frame_paths = extract_frames(video_path)
        print(f"Extracted {len(frame_paths)} frames")

        # Step 2: Detect people in frames
        print("Running person detection...")
        detections = detect_people(frame_paths)
        print(f"Found people in {len(detections)} frames")

        # Step 3: Infer actions (pass frame_paths for VideoMAE)
        print("Analyzing actions...")
        action_data = infer_actions(detections, frame_paths)

        # Step 4: Build structured summary
        summary = build_summary(action_data)
        print(f"Summary: {summary}")

        # Step 5: Generate natural language verdict
        print("Generating verdict with Claude...")
        verdict = summarize_with_claude(summary)

        return verdict

    finally:
        # Always cleanup temporary frames
        if frame_paths:
            cleanup_frames(frame_paths)
            print("Cleaned up temporary frames")
