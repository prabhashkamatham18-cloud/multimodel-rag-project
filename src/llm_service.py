from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request

from src.config import Settings


class OllamaService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def list_models(self) -> list[str]:
        request = urllib.request.Request(f"{self.settings.ollama_base_url}/api/tags")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.settings.ollama_base_url}. Start Ollama and retry."
            ) from exc
        return [str(model["name"]) for model in payload.get("models", [])]

    def verify_model(self, model: str) -> None:
        installed = self.list_models()
        if model not in installed:
            raise RuntimeError(
                f"Model '{model}' is not installed in Ollama. Run: ollama pull {model}"
            )

    def generate(self, prompt: str, images: list[bytes] | None = None) -> str:
        model = self.settings.vision_model if images else self.settings.text_model
        if not model:
            raise RuntimeError("Configure an Ollama vision model before asking about an image.")
        self.verify_model(model)
        payload: dict[str, object] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        if images:
            payload["images"] = [base64.b64encode(image).decode("ascii") for image in images]
        request = urllib.request.Request(
            f"{self.settings.ollama_base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama generation failed: {exc}") from exc
        answer = str(result.get("response", "")).strip()
        if not answer:
            raise RuntimeError("Ollama returned an empty answer.")
        return answer
