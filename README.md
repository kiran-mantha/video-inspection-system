# Video Inspection System

A practical, production-oriented video surveillance analysis system.

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
Claude Vision (Semantic Analysis)
 ↓
Rule Engine
 ↓
SAFE / UNSAFE + Summary
```

## Features

- **Object Detection**: YOLOv8n detects people, weapons (knife, scissors), and other objects
- **Frame Gating**: Reduces costs by skipping LLM calls when no person is detected
- **Claude Vision**: Sends selected frames for semantic understanding
- **Rule Engine**: Deterministic safety classification
- **System Monitoring**: Tracks CPU, RAM, and GPU usage during analysis

## Design Principles

1. **Deterministic before probabilistic** - YOLO runs first
2. **Cheap models before expensive models** - LLM called only when needed
3. **Explain, don't guess** - Claude describes only what's visible
4. **Rules over intuition** - Final decision is rule-based
5. **Human-readable output** - Clear safety verdicts

## Installation

```bash
# Create virtual environment
python -m venv virtual_env
virtual_env\Scripts\activate  # Windows
# source virtual_env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Set your Claude API key:

```bash
set ANTHROPIC_API_KEY=your_api_key_here  # Windows
# export ANTHROPIC_API_KEY=your_api_key_here  # Linux/Mac
```

## Usage

```bash
python test.py
```

Then enter the path to your video file when prompted.

### Example Output

```
======================================================================
VIDEO INSPECTION SYSTEM - TEST
Architecture: YOLO → Frame Gating → Claude Vision → Rule Engine
======================================================================

📁 Video: C:\Users\Downloads\surveillance.mp4

📹 Extracting frames from video...
   Extracted 15 frames
🔍 Running object detection (YOLO)...
   Found people in 12 frames
   Objects: {'person': 12, 'knife': 3}
🚦 Applying frame gating logic...
   Potentially dangerous objects detected - requires analysis
🧠 Analyzing 3 frames with Claude Vision...
   Risk Level: HIGH
⚖️  Applying safety rules...
   Classification: UNSAFE
🧹 Cleaned up temporary frames

======================================================================
📋 FINAL VERDICT:
======================================================================

A person is visible holding what appears to be a knife while moving through the scene.
⚠️ The footage is UNSAFE and requires attention.

⏱️  Total Processing Time: 4.23 seconds
```

## Files

| File | Purpose |
|------|---------|
| `video_inspector.py` | Main orchestration and all detection logic |
| `config.py` | Configuration settings and prompts |
| `utils.py` | Utility functions |
| `test.py` | Test script with system monitoring |
| `system_monitor.py` | CPU/RAM/GPU monitoring |

## Requirements

- Python 3.9+
- OpenCV
- Ultralytics (YOLOv8)
- Anthropic SDK
- psutil (optional, for monitoring)

## License

MIT License
