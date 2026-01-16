# 🎥 Video Inspection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/YOLOv8-Person%20Detection-green.svg" alt="YOLOv8">
  <img src="https://img.shields.io/badge/VideoMAE-Action%20Recognition-purple.svg" alt="VideoMAE">
  <img src="https://img.shields.io/badge/Claude-AI%20Summaries-orange.svg" alt="Claude API">
</p>

A **plain Python** video surveillance analysis system that detects people and their actions using state-of-the-art deep learning models.

---

## ✨ Features

| Feature | Technology | Description |
|---------|------------|-------------|
| **Person Detection** | YOLOv8n | Fast and accurate detection of people in video frames |
| **Action Recognition** | VideoMAE Transformer | 400+ action classes from Kinetics dataset |
| **Natural Language** | Claude API | Converts structured facts to 2-3 sentence summaries |
| **No Framework** | Pure Python | No Django, FastAPI, or web frameworks |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VIDEO INSPECTION PIPELINE                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   📹 Video    ──▶   🖼️ Frame        ──▶   👤 Person Detection    │
│      File           Extraction            (YOLOv8n)                 │
│                     (OpenCV)                                        │
│                                                                     │
│                           │                                         │
│                           ▼                                         │
│                                                                     │
│   📄 Final    ◀──   🤖 Claude API   ◀──   🎬 Action Recognition  │
│      Verdict        (Text Only)           (VideoMAE Transformer)    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/video-inspection-system.git
cd video-inspection-system
```

### 2. Create Virtual Environment

```bash
python -m venv virtual_env

# Windows
.\virtual_env\Scripts\activate

# Linux/macOS
source virtual_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variable

```bash
# Windows PowerShell
$env:ANTHROPIC_API_KEY="your-api-key-here"

# Windows CMD
set ANTHROPIC_API_KEY=your-api-key-here

# Linux/macOS
export ANTHROPIC_API_KEY="your-api-key-here"
```

---

## 🚀 Usage

### Basic Usage

```python
from video_inspector import inspect_video

# Analyze a video file
result = inspect_video("path/to/your/video.mp4")
print(result)
```

### Example Outputs

| Scenario | Output |
|----------|--------|
| No person detected | `"The footage is safe."` |
| Person walking | `"A person was detected in the footage for 15 seconds. They were observed walking through the area."` |
| Person with actions | `"A person appeared in the footage for approximately 30 seconds. The individual was seen dancing and appears to have been moving throughout."` |

---

## 📁 Project Structure

```
video_processing_with_ml/
├── video_inspector.py      # Main module with 6 core functions
├── action_recognizer.py    # VideoMAE transformer integration
├── config.py               # Configuration settings
├── utils.py                # Helper utilities
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🔧 API Reference

### Core Functions

#### `extract_frames(video_path, fps=1)`
Extracts frames from video at specified FPS rate.

```python
frames = extract_frames("video.mp4", fps=1)
# Returns: ['temp_frames/video_frame_00000.png', ...]
```

#### `detect_people(frame_paths)`
Runs YOLOv8n person detection on frames.

```python
detections = detect_people(frames)
# Returns: [{'frame_index': 0, 'timestamp': 0.0, 'bounding_boxes': [...], 'confidences': [...]}]
```

#### `infer_actions(detections, frame_paths)`
Uses VideoMAE to recognize actions.

```python
action_data = infer_actions(detections, frames)
# Returns: {'person_detected': True, 'actions': [{'action': 'walking', 'confidence': 0.85}], ...}
```

#### `build_summary(action_data)`
Creates structured JSON facts.

```python
summary = build_summary(action_data)
# Returns: {'person_present': True, 'detected_actions': ['walking'], ...}
```

#### `summarize_with_claude(summary)`
Converts facts to natural language.

```python
verdict = summarize_with_claude(summary)
# Returns: "A person was detected walking through the area for 10 seconds."
```

#### `inspect_video(video_path)`
Main orchestration function.

```python
result = inspect_video("surveillance.mp4")
# Returns: Complete inspection verdict
```

---

## 🎯 Supported Actions

The VideoMAE model recognizes **400+ action classes** including:

<details>
<summary>Click to expand action categories</summary>

### Movement
- walking, running, jogging, crawling
- jumping, hopping, skipping

### Gestures
- waving, pointing, clapping
- shaking hands, hugging

### Daily Activities
- eating, drinking, cooking
- reading, writing, typing
- using phone, taking photos

### Sports & Exercise
- dancing, exercising, stretching
- playing sports (basketball, soccer, etc.)

### Object Interactions
- opening/closing doors
- carrying objects
- pushing, pulling

</details>

---

## ⚙️ Configuration

Edit `config.py` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `PERSON_CONFIDENCE_THRESHOLD` | 0.5 | Minimum confidence for person detection |
| `DEFAULT_FPS` | 1 | Frames per second for extraction |
| `TOP_K_ACTIONS` | 3 | Number of top actions to report |
| `ACTION_CONFIDENCE_THRESHOLD` | 0.2 | Minimum confidence for actions |

---

## 🔒 Privacy & Security

> **Claude API receives ONLY structured JSON facts — never images or video data.**

```json
// Example data sent to Claude
{
  "person_present": true,
  "visibility_duration": "15 seconds",
  "detected_actions": ["walking", "opening door"],
  "movement": "walking"
}
```

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions are welcome! Please read the contributing guidelines before submitting pull requests.

---

<p align="center">
  Built with ❤️ using YOLOv8, VideoMAE, and Claude
</p>
