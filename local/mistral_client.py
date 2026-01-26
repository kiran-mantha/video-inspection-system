# Mistral Client
# ===============
# Client for the Mistral LLM via Ollama Docker service

import requests
from typing import Optional


class MistralClient:
    """
    Client for Mistral LLM via Ollama.

    Connects to an Ollama Docker container running the Mistral model.
    """

    def __init__(self, api_url: str, model: str = "mistral", timeout: int = 60):
        """
        Initialize the Mistral client.

        Args:
            api_url: URL of the Ollama API endpoint
            model: Model name to use (default: mistral)
            timeout: Request timeout in seconds
        """
        self.api_url = api_url
        self.model = model
        self.timeout = timeout

    def generate_response(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate LLM response based on prompt.

        Args:
            prompt: The user prompt to send
            system_prompt: Optional system prompt for context

        Returns:
            Generated response from Mistral
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }

            if system_prompt:
                payload["system"] = system_prompt

            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama service error: {response.status_code} - {response.text}"
                )

            result = response.json()
            return result.get("response", "")

        except requests.ConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to Ollama service at {self.api_url}. "
                "Make sure the Docker container is running."
            ) from e
        except requests.Timeout as e:
            raise RuntimeError(
                f"Ollama service timeout after {self.timeout} seconds."
            ) from e

    def health_check(self) -> bool:
        """Check if the Ollama service is healthy."""
        try:
            base_url = self.api_url.rsplit("/api", 1)[0]
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
