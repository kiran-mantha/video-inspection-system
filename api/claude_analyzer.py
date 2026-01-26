# Claude Vision Analyzer
# =======================
# API-based vision analysis using Claude

import base64
from pathlib import Path
from typing import Dict, List

from config import CLAUDE_VISION_PROMPT


class ClaudeVisionAnalyzer:
    """
    Claude API-based vision analyzer.

    Uses Claude's vision capability to analyze surveillance frames
    for security-related concerns.
    """

    def __init__(self, api_key: str, model: str):
        """
        Initialize the Claude Vision analyzer.

        Args:
            api_key: Anthropic API key
            model: Claude model to use (e.g., 'claude-sonnet-4-20250514')
        """
        self.api_key = api_key
        self.model = model

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 string."""
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def _parse_response(self, response: str) -> Dict:
        """Parse structured response into standard format."""
        result = {
            "observation": "",
            "objects": "",
            "risk_level": "LOW",
            "raw_response": response,
        }

        lines = response.strip().split("\n")
        for line in lines:
            line_lower = line.lower()
            if line_lower.startswith("observation:"):
                result["observation"] = line.split(":", 1)[1].strip()
            elif line_lower.startswith("objects:"):
                result["objects"] = line.split(":", 1)[1].strip()
            elif line_lower.startswith("risk_level:") or line_lower.startswith(
                "risk level:"
            ):
                level = line.split(":", 1)[1].strip().upper()
                if level in ["LOW", "MEDIUM", "HIGH"]:
                    result["risk_level"] = level

        if not result["observation"]:
            result["observation"] = response

        return result

    def analyze_frames(self, frame_paths: List[str], detection_context: Dict) -> Dict:
        """
        Send frames to Claude Vision for semantic understanding.

        Args:
            frame_paths: Selected frame paths to analyze
            detection_context: Object detection results for context

        Returns:
            Dictionary with observation, objects, risk_level, raw_response
        """
        from anthropic import Anthropic

        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Please set it to use Claude Vision."
            )

        client = Anthropic(api_key=self.api_key)

        # Build content with images
        content = []

        # Add context about YOLO detections
        object_summary = ", ".join(
            f"{count} {name}"
            for name, count in detection_context.get("object_counts", {}).items()
        )
        if object_summary:
            content.append(
                {
                    "type": "text",
                    "text": f"Context: YOLO detected these objects across frames: {object_summary}",
                }
            )

        # Add frames as images
        for i, frame_path in enumerate(frame_paths):
            image_data = self._encode_image_to_base64(frame_path)

            # Determine media type
            ext = Path(frame_path).suffix.lower()
            media_type = "image/png" if ext == ".png" else "image/jpeg"

            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    },
                }
            )

        # Add final instruction
        content.append(
            {
                "type": "text",
                "text": "Analyze these surveillance frame(s) and provide your assessment.",
            }
        )

        # Call Claude Vision
        response = client.messages.create(
            model=self.model,
            max_tokens=500,
            system=CLAUDE_VISION_PROMPT,
            messages=[{"role": "user", "content": content}],
        )

        raw_response = response.content[0].text.strip()

        # Parse and return structured response
        return self._parse_response(raw_response)
