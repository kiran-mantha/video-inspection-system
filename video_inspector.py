# Video Inspection System
# ========================
# Architecture: YOLO Detection → Frame Gating → BLIP API → Rule Engine
#
# Flow:
# 1. Extract frames from video
# 2. Detect objects (people, weapons, etc.) using YOLOv8
# 3. Gate frames (decide if vision model is needed)
# 4. Send frames to remote BLIP API for captioning
# 5. Apply rule engine for final SAFE/UNSAFE decision

import os
import base64
import requests
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import cv2

from config import (
    BLIP_ANALYZE_ENDPOINT,
    BLIP_HEALTH_ENDPOINT,
    BLIP_API_TIMEOUT,
    DANGER_KEYWORDS,
    MEDIUM_RISK_KEYWORDS,
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

    Args:
        frame_paths: List of absolute paths to frame images

    Returns:
        Dictionary with detection results
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

                if cls_id in DETECTION_CLASSES:
                    class_name, is_dangerous = DETECTION_CLASSES[cls_id]
                else:
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

                object_counts[class_name] = object_counts.get(class_name, 0) + 1

                if cls_id == PERSON_CLASS_ID:
                    frame_detections["has_person"] = True

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
    Decide whether to invoke vision model and select best frames.

    Args:
        frame_paths: All extracted frame paths
        detection_result: Output from detect_objects()

    Returns:
        Tuple of (needs_vision, selected_frames, gate_reason)
    """
    frames_with_people = detection_result["frames_with_people"]
    frames_with_danger = detection_result["frames_with_danger"]

    # Rule 1: No person detected → SAFE, skip vision model
    if not frames_with_people:
        return False, [], "No person detected - footage is safe"

    # Rule 2: Dangerous object detected → HIGH priority
    if frames_with_danger:
        priority_indices = list(set(frames_with_danger + frames_with_people[:2]))
        priority_indices = sorted(priority_indices)[:MAX_FRAMES_FOR_VISION]
        selected = [frame_paths[i] for i in priority_indices]
        return (
            True,
            selected,
            "Potentially dangerous objects detected - requires analysis",
        )

    # Rule 3: Person detected, no danger → Normal priority
    if len(frames_with_people) <= MAX_FRAMES_FOR_VISION:
        selected_indices = frames_with_people
    else:
        step = len(frames_with_people) // MAX_FRAMES_FOR_VISION
        selected_indices = frames_with_people[::step][:MAX_FRAMES_FOR_VISION]

    selected = [frame_paths[i] for i in selected_indices]
    return True, selected, "Person detected - analyzing activity"


# ============================================================
# Function 4: BLIP API Analysis (Remote Docker)
# ============================================================


def check_blip_api_health() -> bool:
    """Check if BLIP API server is available."""
    try:
        response = requests.get(BLIP_HEALTH_ENDPOINT, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def analyze_with_blip_api(frame_paths: List[str], detection_context: Dict) -> Dict:
    """
    Send frames to remote BLIP API for captioning.

    Args:
        frame_paths: Selected frame paths to analyze
        detection_context: Object detection results for context

    Returns:
        Dictionary with:
        - captions: List of generated captions
        - combined_caption: Merged description
        - risk_level: LOW/MEDIUM/HIGH based on keywords
    """
    captions = []

    for frame_path in frame_paths:
        # Read and encode image as base64
        with open(frame_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # Send to BLIP API
        try:
            response = requests.post(
                BLIP_ANALYZE_ENDPOINT,
                json={"image": image_data},
                timeout=BLIP_API_TIMEOUT,
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    captions.append(result.get("caption", ""))
                else:
                    print(f"      [WARN] API error: {result.get('error')}")
            else:
                print(f"      [WARN] API returned status {response.status_code}")

        except requests.exceptions.Timeout:
            print(f"      [WARN] API timeout for frame: {frame_path}")
        except requests.exceptions.RequestException as e:
            print(f"      [WARN] API error: {e}")

    # Combine captions
    combined_caption = " ".join(captions) if captions else "Unable to analyze frames"

    # Assess risk based on keywords
    risk_level = assess_risk_from_caption(combined_caption)

    return {
        "captions": captions,
        "combined_caption": combined_caption,
        "risk_level": risk_level,
        "observation": combined_caption,
    }


def assess_risk_from_caption(caption: str) -> str:
    """
    Assess risk level based on keywords in caption.

    Args:
        caption: Generated caption text

    Returns:
        Risk level: LOW, MEDIUM, or HIGH
    """
    caption_lower = caption.lower()

    # Check for danger keywords
    for keyword in DANGER_KEYWORDS:
        if keyword in caption_lower:
            return "HIGH"

    # Check for medium risk keywords
    for keyword in MEDIUM_RISK_KEYWORDS:
        if keyword in caption_lower:
            return "MEDIUM"

    return "LOW"


# ============================================================
# Function 5: Rule Engine (Safety Classification)
# ============================================================


def classify_safety(
    detection_result: Dict, vision_result: Optional[Dict]
) -> Tuple[str, str]:
    """
    Apply rule engine to determine final safety classification.

    Args:
        detection_result: YOLO detection output
        vision_result: BLIP analysis (can be None)

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

    # Rule 3-4: Vision analysis
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
        vision_result: BLIP analysis (can be None)
        safety_level: SAFE/UNSAFE/REVIEW
        safety_explanation: Explanation from rule engine

    Returns:
        Human-readable summary string
    """
    parts = []

    # Add observation from BLIP
    if vision_result and vision_result.get("observation"):
        observation = vision_result["observation"].strip()
        if observation and observation != "Unable to analyze frames":
            observation = observation[0].upper() + observation[1:]
            if not observation.endswith("."):
                observation += "."
            parts.append(observation)

    # Add safety verdict
    if safety_level == SAFETY_SAFE:
        verdict = "The footage is SAFE."
    elif safety_level == SAFETY_UNSAFE:
        verdict = "WARNING: The footage is UNSAFE and requires attention."
    else:
        verdict = "WARNING: The footage requires REVIEW."

    # If no vision analysis, use explanation
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
    3. Gate frames (decide if vision model needed)
    4. Analyze with BLIP API (if needed)
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

        # Step 4: BLIP API (if needed)
        vision_result = None
        if needs_vision:
            # Check API health first
            if check_blip_api_health():
                print(f"[4/6] Sending {len(selected_frames)} frames to BLIP API...")
                vision_result = analyze_with_blip_api(selected_frames, detection_result)
                print(f"      Risk Level: {vision_result['risk_level']}")
                print(f"      Caption: {vision_result['combined_caption']}")
            else:
                print("[4/6] BLIP API not available - skipping vision analysis")
                print(f"      Check server at: {BLIP_ANALYZE_ENDPOINT}")
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
