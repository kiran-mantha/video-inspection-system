# Test script for Video Inspection System
# ========================================

import time

from system_monitor import MetricsCollector, capture_baseline, print_comparison
from video_inspector import inspect_video

# Video path to test (leave empty to prompt for input)
VIDEO_PATH = ""


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

    print("=" * 60)
    print("VIDEO INSPECTION SYSTEM - TEST")
    print("=" * 60)
    print(f"\nAnalyzing: {video_path}")

    # Capture baseline metrics BEFORE starting
    print("\n📊 Capturing baseline system metrics...")
    baseline = capture_baseline(sample_duration=1.0)
    print(
        f"   ✓ Baseline captured: CPU {baseline.cpu_percent:.1f}% | "
        f"RAM {baseline.memory_percent:.1f}%"
    )

    print("\n📈 Starting inspection with system monitoring...\n")

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

        print("\n" + "=" * 60)
        print("VERDICT:")
        print("=" * 60)
        print(f"\n{result}\n")
        print(f"⏱️  Total Processing Time: {elapsed_time:.2f} seconds")

        # Print comparison: baseline vs during inspection
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


if __name__ == "__main__":
    main()
