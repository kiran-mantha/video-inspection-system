# Test script for Video Inspection System
# ========================================

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
    print(f"\nAnalyzing: {video_path}\n")

    try:
        result = inspect_video(video_path)
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
