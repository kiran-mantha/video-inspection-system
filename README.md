# Video Inspection System

A practical, production-oriented video surveillance analysis system using local models.

## Architecture

```
Video
 ↓
Frame Extraction (OpenCV)
 ↓
Object Detection (YOLOv8)
 ↓
Frame Gating (Cost Control)
 ↓
BLIP Vision (Local Captioning)
 ↓
Rule Engine
 ↓
SAFE / UNSAFE + Summary
```

## Features

- **Object Detection**: YOLOv8n detects people, weapons (knife, scissors), and other objects
- **Frame Gating**: Skips vision model when no person is detected
- **BLIP Vision**: Local image captioning (no API required!)
- **Rule Engine**: Keyword-based risk assessment + deterministic safety classification
- **GPU Support**: Automatically uses CUDA if available
- **System Monitoring**: Tracks CPU, RAM, and GPU usage during analysis

## Models Used

| Model | Purpose | Size |
|-------|---------|------|
| `yolov8n.pt` | Object detection | ~6 MB |
| `Salesforce/blip-image-captioning-large` | Image captioning | ~1.5 GB |

## Installation

```bash
# Create virtual environment
python -m venv virtual_env
virtual_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python test.py
```

Enter the path to your video file when prompted.

### Example Output

```
======================================================================
VIDEO INSPECTION SYSTEM - TEST
Architecture: YOLO → Frame Gating → BLIP Vision → Rule Engine
======================================================================

[1/6] Extracting frames from: video.mp4
      Extracted 15 frames
[2/6] Running object detection (YOLO)...
      Found people in 12 frames
      Objects: {'person': 12}
[3/6] Applying frame gating logic...
      Person detected - analyzing activity
[4/6] Analyzing 3 frames with BLIP...
      Risk Level: HIGH
      Caption: a man holding a gun walking down the street
[5/6] Applying safety rules...
      Classification: UNSAFE
[6/6] Cleaned up temporary frames

======================================================================
FINAL VERDICT:
======================================================================

A man holding a gun walking down the street. WARNING: The footage is UNSAFE.
```

## Risk Assessment

BLIP generates captions, and the system checks for keywords:

**HIGH Risk Keywords:**
- gun, pistol, rifle, firearm, weapon, knife, blade, sword
- fighting, attack, violence, blood, injured

**MEDIUM Risk Keywords:**
- running, chasing, arguing, suspicious, hiding

## Files

| File | Purpose |
|------|---------|
| `video_inspector.py` | Main orchestration and detection logic |
| `config.py` | Configuration and keyword lists |
| `utils.py` | Utility functions |
| `test.py` | Test script with system monitoring |
| `system_monitor.py` | CPU/RAM/GPU monitoring |

## License

MIT License
