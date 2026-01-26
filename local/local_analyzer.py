# Local Vision Analyzer
# ======================
# Combines BLIP and Mistral for local vision analysis

from typing import Dict, List

from local.blip_client import BLIPClient
from local.mistral_client import MistralClient


# Security analysis prompt for Mistral - mirrors Claude's prompt
MISTRAL_SYSTEM_PROMPT = """You are a security footage analyst. You will receive image descriptions from a vision model analyzing surveillance footage.

TASK:
1. Based on the image descriptions, provide a concise security assessment (1-2 sentences)
2. Focus on: people present, their actions, and any objects they are interacting with
3. Note any potentially dangerous items (weapons, suspicious objects)
4. Flag if anyone is in a restricted area or without required safety equipment

RULES:
- Only describe what was reported in the image descriptions
- Do NOT speculate about intent or identity
- Do NOT make assumptions about what happened before or after
- Be factual and objective
- If weapons or dangerous items are mentioned, explicitly note them

OUTPUT FORMAT:
Observation: [1-2 sentences describing the scene based on the image descriptions]
Objects: [list key objects detected, especially any weapons or safety violations]
Risk_Level: [LOW / MEDIUM / HIGH]"""


class LocalVisionAnalyzer:
    """
    Local model-based vision analyzer using BLIP + Mistral.

    Pipeline:
    1. BLIP analyzes each frame and generates captions
    2. Mistral processes the captions with security context
    3. Returns structured security assessment
    """

    def __init__(self, blip_url: str, mistral_url: str, mistral_model: str):
        """
        Initialize the local vision analyzer.

        Args:
            blip_url: URL of the BLIP API endpoint
            mistral_url: URL of the Ollama/Mistral API endpoint
            mistral_model: Mistral model name to use
        """
        self.blip = BLIPClient(blip_url)
        self.mistral = MistralClient(mistral_url, mistral_model)

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
        Analyze frames using local BLIP + Mistral models.

        Args:
            frame_paths: Selected frame paths to analyze
            detection_context: Object detection results for context

        Returns:
            Dictionary with observation, objects, risk_level, raw_response
        """
        # Step 1: Get BLIP captions for each frame
        captions = []
        for i, frame_path in enumerate(frame_paths):
            try:
                caption = self.blip.analyze_image(
                    frame_path,
                    question="What is happening in this surveillance image? Describe any people, their actions, and objects.",
                )
                captions.append(f"Frame {i + 1}: {caption}")
            except Exception as e:
                captions.append(f"Frame {i + 1}: [Error analyzing: {str(e)}]")

        # Step 2: Build prompt with YOLO context and BLIP captions
        object_summary = ", ".join(
            f"{count} {name}"
            for name, count in detection_context.get("object_counts", {}).items()
        )

        prompt_parts = []

        if object_summary:
            prompt_parts.append(f"YOLO object detection found: {object_summary}")

        prompt_parts.append("\nImage descriptions from vision model:")
        prompt_parts.extend(captions)
        prompt_parts.append(
            "\nBased on the above information, provide your security assessment."
        )

        prompt = "\n".join(prompt_parts)

        # Step 3: Get Mistral analysis
        try:
            response = self.mistral.generate_response(
                prompt=prompt,
                system_prompt=MISTRAL_SYSTEM_PROMPT,
            )
        except Exception as e:
            return {
                "observation": f"Error analyzing frames: {str(e)}",
                "objects": "",
                "risk_level": "LOW",
                "raw_response": f"Error: {str(e)}",
            }

        # Step 4: Parse and return structured response
        return self._parse_response(response)

    def health_check(self) -> Dict[str, bool]:
        """Check if both BLIP and Mistral services are healthy."""
        return {
            "blip": self.blip.health_check(),
            "mistral": self.mistral.health_check(),
        }
