# Action Recognition using VideoMAE Transformer
# This module handles sophisticated action recognition using a pre-trained
# VideoMAE model fine-tuned on the Kinetics-400 dataset.

from typing import List, Dict, Tuple
import numpy as np

# Lazy imports to avoid loading heavy models at module import time
_model = None
_processor = None


def _load_model():
    """
    Lazily load the VideoMAE model and processor.
    This avoids loading the model until it's actually needed.
    """
    global _model, _processor

    if _model is None or _processor is None:
        from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
        from config import VIDEOMAE_MODEL

        print(f"Loading VideoMAE model: {VIDEOMAE_MODEL}...")
        _processor = VideoMAEImageProcessor.from_pretrained(VIDEOMAE_MODEL)
        _model = VideoMAEForVideoClassification.from_pretrained(VIDEOMAE_MODEL)
        _model.eval()  # Set to evaluation mode
        print("VideoMAE model loaded successfully.")

    return _model, _processor


def sample_frames_for_action_recognition(
    frame_paths: List[str], num_frames: int = 16
) -> List[str]:
    """
    Sample frames uniformly for action recognition.
    VideoMAE expects a fixed number of frames (default 16).

    Args:
        frame_paths: List of all extracted frame paths
        num_frames: Number of frames to sample (default 16 for VideoMAE)

    Returns:
        List of sampled frame paths
    """
    if len(frame_paths) <= num_frames:
        # If we have fewer frames than needed, repeat frames
        indices = np.linspace(0, len(frame_paths) - 1, num_frames, dtype=int)
    else:
        # Uniformly sample frames across the video
        indices = np.linspace(0, len(frame_paths) - 1, num_frames, dtype=int)

    return [frame_paths[i] for i in indices]


def load_frames_as_video(frame_paths: List[str]) -> np.ndarray:
    """
    Load frame images and stack them as a video tensor.

    Args:
        frame_paths: List of frame image paths (should be 16 frames)

    Returns:
        Numpy array of shape (num_frames, height, width, 3)
    """
    from PIL import Image

    frames = []
    for path in frame_paths:
        img = Image.open(path).convert("RGB")
        frames.append(np.array(img))

    return np.stack(frames, axis=0)


def recognize_actions(
    frame_paths: List[str], top_k: int = 3, confidence_threshold: float = 0.2
) -> List[Dict]:
    """
    Recognize actions in video frames using VideoMAE transformer.

    Args:
        frame_paths: List of frame image paths
        top_k: Number of top predictions to return
        confidence_threshold: Minimum confidence for an action

    Returns:
        List of dictionaries with 'action' and 'confidence' keys
    """
    import torch
    from config import ACTION_RECOGNITION_NUM_FRAMES

    if not frame_paths:
        return []

    # Load model lazily
    model, processor = _load_model()

    # Sample frames for action recognition
    sampled_paths = sample_frames_for_action_recognition(
        frame_paths, ACTION_RECOGNITION_NUM_FRAMES
    )

    # Load frames as video array
    video_frames = load_frames_as_video(sampled_paths)

    # Process video for model input
    # VideoMAE expects list of frames as input
    inputs = processor(list(video_frames), return_tensors="pt")

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get probabilities via softmax
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    # Get top-k predictions
    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))

    # Convert to action labels
    actions = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        if prob >= confidence_threshold:
            label = model.config.id2label[idx]
            # Clean up label (Kinetics labels often have underscores)
            label = label.replace("_", " ")
            actions.append({"action": label, "confidence": round(prob, 3)})

    return actions


def get_kinetics_labels() -> Dict[int, str]:
    """
    Get the mapping of class IDs to action labels for Kinetics-400.

    Returns:
        Dictionary mapping class ID to action label
    """
    model, _ = _load_model()
    return model.config.id2label
