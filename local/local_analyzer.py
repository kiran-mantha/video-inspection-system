# Local Vision Analyzer
# ======================
# Uses LLaVA from Ollama for vision analysis

from typing import Dict, List

from config import ANALYSIS_SYSTEM_PROMPT
from local.ollama_vision_client import OllamaVisionClient


class LocalVisionAnalyzer:
    """
    Local model-based vision analyzer using LLaVA via Ollama.

    Pipeline:
    - LLaVA analyzes frames and generates structured output directly
    - Similar to Claude Vision API approach (single model, direct output)
    """

    def __init__(self, ollama_url: str, vision_model: str):
        """
        Initialize the local vision analyzer.

        Args:
            ollama_url: URL of the Ollama API endpoint
            vision_model: Vision model name (e.g., 'llava:13b')
        """
        self.llava = OllamaVisionClient(ollama_url, vision_model, timeout=120)

    def analyze_frames(self, frame_paths: List[str], detection_context: Dict) -> Dict:
        """
        Analyze frames using LLaVA with structured output.

        Args:
            frame_paths: Selected frame paths to analyze
            detection_context: Object detection results for context

        Returns:
            Dictionary with observation, objects, risk_level, raw_response
        """
        # Build context from YOLO detections
        object_summary = ", ".join(
            f"{count} {name}"
            for name, count in detection_context.get("object_counts", {}).items()
        )

        # Build comprehensive prompt for LLaVA
        prompt_parts = []

        if object_summary:
            prompt_parts.append(
                f"Context: YOLO object detection found these objects: {object_summary}"
            )

        prompt_parts.append("\n" + ANALYSIS_SYSTEM_PROMPT)
        prompt_parts.append(
            "\nAnalyze these surveillance frames and provide your security assessment "
            "in the exact JSON format specified above."
        )

        prompt = "\n".join(prompt_parts)

        # Analyze first frame with LLaVA
        try:
            response = self.llava.analyze_image(frame_paths[0], prompt)
        except Exception as e:
            return {
                "observation": f"Error analyzing frames: {str(e)}",
                "objects": "",
                "risk_level": "LOW",
                "raw_response": f"Error: {str(e)}",
            }

        # Parse and return structured response
        return self._parse_response(response)

    def _parse_response(self, response: str) -> Dict:
        """
        Parse structured LLM response into standard format.

        Args:
            response: Raw response from LLM

        Returns:
            Parsed dictionary with observation, objects, risk_level, raw_response
        """
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

    def health_check(self) -> Dict[str, bool]:
        """Check if LLaVA service is healthy."""
        return {"llava": self.llava.health_check()}
