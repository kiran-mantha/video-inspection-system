# Test script for Video Inspection System
# ========================================
# Tests the architecture: YOLO → Frame Gating → Vision Analysis → Rule Engine
#
# Configure the settings below before running.
# These override the defaults in config.py.

import os
import time

# ============================================================
# TEST CONFIGURATION (edit these before running)
# ============================================================

# Video path to test (leave empty to prompt for input)
VIDEO_PATH = ""

# Model backend: set to "true" to use local LLaVA, "false" for Claude API
USE_LOCAL_MODELS = "false"

# Ollama settings (used when USE_LOCAL_MODELS is "true")
OLLAMA_IP = "localhost"
OLLAMA_PORT = "11434"
OLLAMA_VISION_MODEL = "llava:13b"

# Claude API key (used when USE_LOCAL_MODELS is "false")
ANTHROPIC_API_KEY = ""

# ============================================================
# Apply overrides to environment BEFORE importing config
# ============================================================

os.environ["USE_LOCAL_MODELS"] = USE_LOCAL_MODELS
os.environ["OLLAMA_IP"] = OLLAMA_IP
os.environ["OLLAMA_PORT"] = OLLAMA_PORT
os.environ["OLLAMA_VISION_MODEL"] = OLLAMA_VISION_MODEL
if ANTHROPIC_API_KEY:
    os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

# ============================================================
# Imports (AFTER env overrides)
# ============================================================

from system_monitor import MetricsCollector, capture_baseline, print_comparison
from video_inspector import inspect_video


def main():
    """Run video inspection and display results."""
    # Get video path from constant or user input
    if VIDEO_PATH.strip():
        video_path = VIDEO_PATH.strip()
    else:
        print('Enter video path (e.g., "C:\\Users\\Downloads\\video.mp4"):')
        video_path = input().strip()
        # Remove surrounding quotes if present
        if (video_path.startswith('"') and video_path.endswith('"')) or (
            video_path.startswith("'") and video_path.endswith("'")
        ):
            video_path = video_path[1:-1]

    if not video_path:
        print("❌ Error: No video path provided")
        return

    # Show active configuration
    backend = "Local LLaVA (Ollama)" if USE_LOCAL_MODELS == "true" else "Claude API"
    print("=" * 70)
    print("VIDEO INSPECTION SYSTEM - TEST")
    print(f"Backend: {backend}")
    if USE_LOCAL_MODELS == "true":
        print(f"Ollama: http://{OLLAMA_IP}:{OLLAMA_PORT}")
        print(f"Vision Model: {OLLAMA_VISION_MODEL}")
    print("=" * 70)
    print(f"\n📁 Video: {video_path}")

    # Capture baseline metrics BEFORE starting
    print("\n📊 Capturing baseline system metrics...")
    baseline = capture_baseline(sample_duration=1.0)
    print(
        f"   ✓ Baseline: CPU {baseline.cpu_percent:.1f}% | RAM {baseline.memory_percent:.1f}%"
    )

    print("\n" + "-" * 70)

    # Initialize metrics collector
    metrics_collector = MetricsCollector()

    try:
        # Start monitoring
        metrics_collector.start()
        start_time = time.time()

        # Run video inspection
        result = inspect_video(video_path)

        elapsed_time = time.time() - start_time

        # Stop monitoring
        metrics_collector.stop()

        print("\n" + "=" * 70)
        print("📋 FINAL VERDICT:")
        print("=" * 70)
        print(f"\n{result}\n")
        print(f"⏱️  Total Processing Time: {elapsed_time:.2f} seconds")

        # Print system metrics comparison
        summary = metrics_collector.get_summary()
        print_comparison(baseline, summary)

    except FileNotFoundError as e:
        metrics_collector.stop()
        print(f"\n❌ Error: Video file not found\n   {e}")
    except ValueError as e:
        metrics_collector.stop()
        print(f"\n❌ Error: {e}")
    except Exception as e:
        metrics_collector.stop()
        print(f"\n❌ Unexpected error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
