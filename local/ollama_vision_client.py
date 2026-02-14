# Ollama Vision Client
# =====================
# Client for Ollama vision models (LLaVA)

import requests
import base64
from typing import Optional


class OllamaVisionClient:
    """
    Client for Ollama vision models (LLaVA).

    Connects to Ollama API to analyze images using vision-language models.
    """

    def __init__(self, api_url: str, model: str, timeout: int = 120):
        """
        Initialize the Ollama vision client.

        Args:
            api_url: URL of the Ollama API endpoint (e.g., http://localhost:11434/api/generate)
            model: Model name to use (e.g., 'llava:13b')
            timeout: Request timeout in seconds (default 120 for vision models)
        """
        self.api_url = api_url
        self.model = model
        self.timeout = timeout

    def analyze_image(self, image_path: str, prompt: str) -> str:
        """
        Send image to LLaVA via Ollama and return response.

        Args:
            image_path: Path to the image file
            prompt: Instruction prompt for the model

        Returns:
            Model response text

        Raises:
            ConnectionError: If the Ollama service is not reachable
            RuntimeError: If the Ollama service returns an error
        """
        try:
            # Encode image to base64
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()

            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [image_data],
                "stream": False,
            }

            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama error: {response.status_code} - {response.text}"
                )

            return response.json().get("response", "")

        except requests.ConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to Ollama service at {self.api_url}. "
                "Make sure Ollama is running."
            ) from e
        except requests.Timeout as e:
            raise RuntimeError(
                f"Ollama service timeout after {self.timeout} seconds."
            ) from e

    def health_check(self) -> bool:
        """Check if Ollama service is healthy and model is available."""
        try:
            # Check if model is available
            list_url = self.api_url.replace("/api/generate", "/api/tags")
            response = requests.get(list_url, timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(self.model in m.get("name", "") for m in models)
            return False
        except Exception:
            return False
