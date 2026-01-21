# Utility functions for video inspection system
import os
from pathlib import Path
from typing import List

from config import TEMP_FRAMES_DIR


def ensure_temp_dir() -> Path:
    """
    Create the temporary directory for frames if it doesn't exist.

    Returns:
        Path: The path to the temporary frames directory
    """
    TEMP_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    return TEMP_FRAMES_DIR


def cleanup_frames(frame_paths: List[str]) -> None:
    """
    Remove temporary frame files after processing.

    Args:
        frame_paths: List of absolute paths to frame images to delete
    """
    for path in frame_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as e:
            print(f"Warning: Could not remove frame {path}: {e}")

    # Try to remove the temp directory if empty
    try:
        if TEMP_FRAMES_DIR.exists() and not any(TEMP_FRAMES_DIR.iterdir()):
            TEMP_FRAMES_DIR.rmdir()
    except OSError:
        pass


def calculate_timestamp(frame_index: int, fps: float) -> float:
    """
    Convert frame index to timestamp in seconds.

    Args:
        frame_index: Zero-based index of the frame
        fps: Frames per second used during extraction

    Returns:
        Timestamp in seconds (float)
    """
    return frame_index / fps if fps > 0 else 0.0


def get_video_duration(frame_count: int, fps: float) -> float:
    """
    Calculate total video duration from frame count and FPS.

    Args:
        frame_count: Total number of frames extracted
        fps: Frames per second used during extraction

    Returns:
        Duration in seconds
    """
    return frame_count / fps if fps > 0 else 0.0


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Human-readable duration string
    """
    if seconds < 1:
        return "less than a second"
    elif seconds < 60:
        return f"{int(seconds)} second{'s' if seconds >= 2 else ''}"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        parts = []
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
        if remaining_seconds > 0:
            parts.append(
                f"{remaining_seconds} second{'s' if remaining_seconds > 1 else ''}"
            )
        return " ".join(parts)
