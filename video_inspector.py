# Video Inspection System
# ========================
# Architecture: YOLO Detection → Frame Gating → Claude Vision → Rule Engine
#
# Flow:
# 1. Extract frames from video
# 2. Detect objects (people, weapons, etc.) using YOLOv8
# 3. Gate frames (decide if LLM is needed)
# 4. Send selected frames to Claude Vision for semantic understanding
# 5. Apply rule engine for final SAFE/UNSAFE decision

import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import cv2

from config import (
    USE_LOCAL_MODELS,
    YOLO_MODEL_PATH,
    DETECTION_CONFIDENCE_THRESHOLD,
    DETECTION_CLASSES,
    DANGEROUS_CLASS_IDS,
    PERSON_CLASS_ID,
    DEFAULT_FPS,
    MAX_FRAMES,
    MAX_FRAMES_FOR_VISION,
    SAFETY_SAFE,
    SAFETY_UNSAFE,
    SAFETY_REVIEW,
)
from utils import (
    ensure_temp_dir,
    cleanup_frames,
    calculate_timestamp,
)


# ============================================================
# Function 1: Extract Frames from Video
# ============================================================


def extract_frames(video_path: str, fps: int = DEFAULT_FPS) -> List[str]:
    """
    Extract frames from a video file at the specified FPS rate.

    Args:
        video_path: Absolute path to the video file
        fps: Target frames per second to extract (default: 1)

    Returns:
        List of absolute paths to extracted frame images

    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video cannot be opened or has no frames
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")

    # Calculate frame interval
    frame_interval = max(1, int(video_fps / fps))

    temp_dir = ensure_temp_dir()
    video_name = Path(video_path).stem

    frame_paths = []
    frame_index = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            frame_filename = f"{video_name}_frame_{extracted_count:05d}.png"
            frame_path = str(temp_dir / frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            extracted_count += 1

            # Limit total frames
            if extracted_count >= MAX_FRAMES:
                break

        frame_index += 1

    cap.release()
    return frame_paths


# ============================================================
# Function 2: Detect Objects in Frames (YOLO)
# ============================================================


def detect_objects(frame_paths: List[str]) -> Dict:
    """
    Run YOLOv8 object detection on extracted frames.

    Detects people, weapons, and other relevant objects.

    Args:
        frame_paths: List of absolute paths to frame images

    Returns:
        Dictionary containing:
        - frames_with_people: List of frame indices with people
        - frames_with_danger: List of frame indices with dangerous objects
        - all_detections: Detailed detections per frame
        - summary: Object counts and types
    """
    from ultralytics import YOLO

    model = YOLO(YOLO_MODEL_PATH)

    all_detections = []
    frames_with_people = []
    frames_with_danger = []
    object_counts = {}

    for idx, frame_path in enumerate(frame_paths):
        results = model(frame_path, verbose=False)

        frame_detections = {
            "frame_index": idx,
            "timestamp": calculate_timestamp(idx, DEFAULT_FPS),
            "objects": [],
            "has_person": False,
            "has_danger": False,
        }

        for result in results:
            boxes = result.boxes

            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if confidence < DETECTION_CONFIDENCE_THRESHOLD:
                    continue

                # Check if this class is in our detection list
                if cls_id in DETECTION_CLASSES:
                    class_name, is_dangerous = DETECTION_CLASSES[cls_id]
                else:
                    # Include any detected class
                    class_name = model.names.get(cls_id, f"class_{cls_id}")
                    is_dangerous = cls_id in DANGEROUS_CLASS_IDS

                xyxy = box.xyxy[0].tolist()

                frame_detections["objects"].append(
                    {
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": round(confidence, 3),
                        "bbox": [round(c, 2) for c in xyxy],
                        "is_dangerous": is_dangerous,
                    }
                )

                # Track counts
                object_counts[class_name] = object_counts.get(class_name, 0) + 1

                # Track person presence
                if cls_id == PERSON_CLASS_ID:
                    frame_detections["has_person"] = True

                # Track dangerous objects
                if is_dangerous:
                    frame_detections["has_danger"] = True

        if frame_detections["has_person"]:
            frames_with_people.append(idx)

        if frame_detections["has_danger"]:
            frames_with_danger.append(idx)

        all_detections.append(frame_detections)

    return {
        "frames_with_people": frames_with_people,
        "frames_with_danger": frames_with_danger,
        "all_detections": all_detections,
        "object_counts": object_counts,
        "total_frames": len(frame_paths),
    }


# ============================================================
# Function 3: Frame Gating Logic
# ============================================================


def gate_frames(
    frame_paths: List[str], detection_result: Dict
) -> Tuple[bool, List[str], str]:
    """
    Decide whether to invoke Claude Vision and select best frames.

    Implements cost/accuracy control by skipping LLM for safe videos.

    Args:
        frame_paths: All extracted frame paths
        detection_result: Output from detect_objects()

    Returns:
        Tuple of:
        - needs_vision: Boolean - should we call Claude Vision?
        - selected_frames: List of frame paths to send to Claude
        - gate_reason: Explanation of the decision
    """
    frames_with_people = detection_result["frames_with_people"]
    frames_with_danger = detection_result["frames_with_danger"]

    # Rule 1: No person detected → SAFE, skip LLM
    if not frames_with_people:
        return False, [], "No person detected - footage is safe"

    # Rule 2: Dangerous object detected → HIGH priority, use LLM
    if frames_with_danger:
        # Select frames with danger + some with people
        priority_indices = list(set(frames_with_danger + frames_with_people[:2]))
        priority_indices = sorted(priority_indices)[:MAX_FRAMES_FOR_VISION]
        selected = [frame_paths[i] for i in priority_indices]
        return (
            True,
            selected,
            "Potentially dangerous objects detected - requires analysis",
        )

    # Rule 3: Person detected, no danger → Normal priority
    # Select frames uniformly across video
    if len(frames_with_people) <= MAX_FRAMES_FOR_VISION:
        selected_indices = frames_with_people
    else:
        # Sample uniformly
        step = len(frames_with_people) // MAX_FRAMES_FOR_VISION
        selected_indices = frames_with_people[::step][:MAX_FRAMES_FOR_VISION]

    selected = [frame_paths[i] for i in selected_indices]
    return True, selected, "Person detected - analyzing activity"


# ============================================================
# Function 4: Vision Analysis (API or Local)
# ============================================================


def analyze_with_vision(frame_paths: List[str], detection_context: Dict) -> Dict:
    """
    Analyze frames using the configured vision analyzer.

    Uses Claude Vision API or local BLIP + Mistral models based on
    the USE_LOCAL_MODELS configuration setting.

    Args:
        frame_paths: Selected frame paths to analyze
        detection_context: Object detection results for context

    Returns:
        Dictionary with:
        - observation: What was observed
        - objects: Key objects identified
        - risk_level: LOW/MEDIUM/HIGH
        - raw_response: Full response
    """
    if USE_LOCAL_MODELS:
        from local import LocalVisionAnalyzer
        from config import BLIP_API_URL, MISTRAL_API_URL, MISTRAL_MODEL

        analyzer = LocalVisionAnalyzer(
            blip_url=BLIP_API_URL,
            mistral_url=MISTRAL_API_URL,
            mistral_model=MISTRAL_MODEL,
        )
    else:
        from api import ClaudeVisionAnalyzer
        from config import ANTHROPIC_API_KEY, CLAUDE_MODEL

        analyzer = ClaudeVisionAnalyzer(
            api_key=ANTHROPIC_API_KEY,
            model=CLAUDE_MODEL,
        )

    return analyzer.analyze_frames(frame_paths, detection_context)


# ============================================================
# Function 5: Rule Engine (Safety Classification)
# ============================================================


def classify_safety(
    detection_result: Dict, vision_result: Optional[Dict]
) -> Tuple[str, str]:
    """
    Apply rule engine to determine final safety classification.

    Rules:
    1. No person detected → SAFE
    2. Dangerous object detected by YOLO → UNSAFE
    3. Claude reports HIGH risk → UNSAFE
    4. Claude reports MEDIUM risk → REVIEW
    5. Otherwise → SAFE

    Args:
        detection_result: YOLO detection output
        vision_result: Claude Vision analysis (can be None)

    Returns:
        Tuple of (safety_level, explanation)
    """
    frames_with_people = detection_result.get("frames_with_people", [])
    frames_with_danger = detection_result.get("frames_with_danger", [])
    object_counts = detection_result.get("object_counts", {})

    # Rule 1: No person detected
    if not frames_with_people:
        return SAFETY_SAFE, "No person detected in the footage."

    # Rule 2: Dangerous object detected by YOLO
    if frames_with_danger:
        for cls_id in DANGEROUS_CLASS_IDS:
            if cls_id in DETECTION_CLASSES:
                name = DETECTION_CLASSES[cls_id][0]
                if name in object_counts:
                    return SAFETY_UNSAFE, f"Dangerous object detected: {name}"

    # Rule 3-4: Claude Vision analysis
    if vision_result:
        risk_level = vision_result.get("risk_level", "LOW")

        if risk_level == "HIGH":
            return SAFETY_UNSAFE, vision_result.get(
                "observation", "High risk activity detected."
            )

        if risk_level == "MEDIUM":
            return SAFETY_REVIEW, vision_result.get(
                "observation", "Activity requires review."
            )

    # Rule 5: Default - person present but no danger
    return SAFETY_SAFE, "Person detected performing normal activity."


# ============================================================
# Function 6: Build Final Summary
# ============================================================


def build_final_summary(
    detection_result: Dict,
    vision_result: Optional[Dict],
    safety_level: str,
    safety_explanation: str,
) -> str:
    """
    Build the final human-readable summary.

    Args:
        detection_result: YOLO detection output
        vision_result: Claude Vision analysis (can be None)
        safety_level: SAFE/UNSAFE/REVIEW
        safety_explanation: Explanation from rule engine

    Returns:
        Human-readable summary string
    """
    parts = []

    # Add observation from Claude Vision
    if vision_result and vision_result.get("observation"):
        parts.append(vision_result["observation"])

    # Add safety verdict
    if safety_level == SAFETY_SAFE:
        verdict = "The footage is SAFE."
    elif safety_level == SAFETY_UNSAFE:
        verdict = "⚠️ The footage is UNSAFE and requires attention."
    else:
        verdict = "⚠️ The footage requires REVIEW."

    # If no Claude analysis, use explanation
    if not parts:
        parts.append(safety_explanation)

    parts.append(verdict)

    return " ".join(parts)


# ============================================================
# Function 7: Main Orchestration Function
# ============================================================


def inspect_video(video_path: str) -> str:
    """
    Main function to inspect a video for safety.

    Architecture:
    1. Extract frames from video
    2. Detect objects using YOLOv8
    3. Gate frames (decide if LLM needed)
    4. Analyze with Claude Vision (if needed)
    5. Apply rule engine for safety classification
    6. Generate final summary

    Args:
        video_path: Absolute path to the video file

    Returns:
        Human-readable safety verdict
    """
    frame_paths = []

    try:
        # Step 1: Extract frames
        print(f"[1/6] Extracting frames from: {video_path}")
        frame_paths = extract_frames(video_path)
        print(f"      Extracted {len(frame_paths)} frames")

        # Step 2: Detect objects with YOLO
        print("[2/6] Running object detection (YOLO)...")
        detection_result = detect_objects(frame_paths)
        print(
            f"      Found people in {len(detection_result['frames_with_people'])} frames"
        )
        if detection_result["object_counts"]:
            print(f"      Objects: {detection_result['object_counts']}")

        # Step 3: Frame gating
        print("[3/6] Applying frame gating logic...")
        needs_vision, selected_frames, gate_reason = gate_frames(
            frame_paths, detection_result
        )
        print(f"      {gate_reason}")

        # Step 4: Vision Analysis (if needed)
        vision_result = None
        if needs_vision:
            backend = (
                "local models (BLIP + Mistral)" if USE_LOCAL_MODELS else "Claude Vision"
            )
            print(f"[4/6] Analyzing {len(selected_frames)} frames with {backend}...")
            vision_result = analyze_with_vision(selected_frames, detection_result)
            print(f"      Risk Level: {vision_result['risk_level']}")
        else:
            print("[4/6] Skipping vision analysis (not needed)")

        # Step 5: Rule engine
        print("[5/6] Applying safety rules...")
        safety_level, safety_explanation = classify_safety(
            detection_result, vision_result
        )
        print(f"      Classification: {safety_level}")

        # Step 6: Build final summary
        summary = build_final_summary(
            detection_result, vision_result, safety_level, safety_explanation
        )

        return summary

    finally:
        # Cleanup temporary frames
        if frame_paths:
            cleanup_frames(frame_paths)
            print("[6/6] Cleaned up temporary frames")
