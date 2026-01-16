# Test script for Video Inspection System
# ========================================

from video_inspector import inspect_video

# Video path to test
VIDEO_PATH = "your video input"


def main():
    """Run video inspection and display results."""
    print("=" * 60)
    print("VIDEO INSPECTION SYSTEM - TEST")
    print("=" * 60)
    print(f"\nAnalyzing: {VIDEO_PATH}\n")

    try:
        result = inspect_video(VIDEO_PATH)
        print("\n" + "=" * 60)
        print("VERDICT:")
        print("=" * 60)
        print(f"\n{result}\n")
    except FileNotFoundError as e:
        print(f"\n❌ Error: Video file not found\n   {e}")
    except ValueError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
