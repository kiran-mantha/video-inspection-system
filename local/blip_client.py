# BLIP Client
# ============
# Client for the BLIP image captioning Docker service

import base64
import requests
from typing import Optional


class BLIPClient:
    """
    Client for BLIP image captioning service.

    Connects to a Docker container running BLIP for image analysis.
    """

    def __init__(self, api_url: str, timeout: int = 30):
        """
        Initialize the BLIP client.

        Args:
            api_url: URL of the BLIP API endpoint (e.g., http://localhost:5050/analyze)
            timeout: Request timeout in seconds
        """
        self.api_url = api_url
        self.timeout = timeout

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 string."""
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def analyze_image(self, image_path: str, question: Optional[str] = None) -> str:
        """
        Send image to BLIP and return caption/answer.

        Args:
            image_path: Path to the image file
            question: Optional question for VQA mode

        Returns:
            Caption or answer from BLIP

        Raises:
            ConnectionError: If the BLIP service is not reachable
            RuntimeError: If the BLIP service returns an error
        """
        try:
            image_data = self._encode_image_to_base64(image_path)

            payload = {
                "image": image_data,
            }

            if question:
                payload["question"] = question

            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"BLIP service error: {response.status_code} - {response.text}"
                )

            result = response.json()
            return result.get("caption", result.get("answer", ""))

        except requests.ConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to BLIP service at {self.api_url}. "
                "Make sure the Docker container is running."
            ) from e
        except requests.Timeout as e:
            raise RuntimeError(
                f"BLIP service timeout after {self.timeout} seconds."
            ) from e

    def health_check(self) -> bool:
        """Check if the BLIP service is healthy."""
        try:
            base_url = self.api_url.rsplit("/", 1)[0]
            response = requests.get(f"{base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
