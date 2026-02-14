# Video Inspection System

A practical, production-oriented video surveillance analysis system with support for both cloud (Claude API) and local (LLaVA via Ollama) vision models.

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
Vision Analysis (Claude API or LLaVA)
 ↓
Rule Engine
 ↓
SAFE / UNSAFE + Summary
```

## Features

- **Object Detection**: Custom YOLOv8 model detects people, weapons (gun, knife), fire, smoke, and violence
- **Frame Gating**: Reduces costs by skipping LLM calls when no person is detected
- **Dual Vision Backend**: Claude Vision API or local LLaVA via Ollama
- **Rule Engine**: Deterministic safety classification
- **System Monitoring**: Tracks CPU, RAM, and GPU usage during analysis

## Installation

```bash
# Create virtual environment
python -m venv virtual_env
virtual_env\Scripts\activate  # Windows
# source virtual_env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Running

All configuration is done in `test.py`. Open the file and edit the settings at the top:

```python
# Video path to test (leave empty to prompt for input)
VIDEO_PATH = ""

# Model backend: "true" for local LLaVA, "false" for Claude API
USE_LOCAL_MODELS = "true"

# Ollama settings (used when USE_LOCAL_MODELS is "true")
OLLAMA_IP = "localhost"
OLLAMA_PORT = "11434"
OLLAMA_VISION_MODEL = "llava:13b"

# Claude API key (used when USE_LOCAL_MODELS is "false")
ANTHROPIC_API_KEY = ""
```

Then run:

```bash
python test.py
```

### Using Local LLaVA (Ollama)

1. Build and run the Ollama Docker container:

```bash
docker build -t ollama-llava docker/ollama/
docker run -d -p 11434:11434 --name ollama-llava ollama-llava
```

2. In `test.py`, set:

```python
USE_LOCAL_MODELS = "true"
OLLAMA_IP = "localhost"
OLLAMA_PORT = "11434"
```

3. Run `python test.py`

### Using Claude API

1. In `test.py`, set:

```python
USE_LOCAL_MODELS = "false"
ANTHROPIC_API_KEY = "your_api_key_here"
```

2. Run `python test.py`

## Files

| File | Purpose |
|------|---------|
| `test.py` | Entry point — configure and run here |
| `video_inspector.py` | Main orchestration and all detection logic |
| `config.py` | Default configuration settings and prompts |
| `utils.py` | Utility functions |
| `system_monitor.py` | CPU/RAM/GPU monitoring |
| `local/` | Local model module (LLaVA via Ollama) |
| `docker/ollama/` | Dockerfile for Ollama with LLaVA |

## Requirements

- Python 3.9+
- OpenCV
- Ultralytics (YOLOv8)
- Anthropic SDK (for Claude API backend)
- psutil (optional, for monitoring)
- Docker (for local LLaVA backend)

## License

MIT License
