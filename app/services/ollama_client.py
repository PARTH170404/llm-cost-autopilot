"""
app/services/ollama_client.py
Unified Ollama client with retry logic and standardized response objects.
"""

import time
import logging
from typing import Optional
import httpx
from app.models.schemas import LLMResponse, ModelConfig
from app.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Unified client for sending requests to any Ollama-hosted model.
    Returns standardized LLMResponse objects regardless of model.
    """

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        cfg = registry.ollama_config
        self.base_url = cfg.get("base_url", "http://localhost:11434")
        self.timeout = cfg.get("timeout", 120)
        self.retry_attempts = cfg.get("retry_attempts", 3)
        self.retry_delay = cfg.get("retry_delay", 2)

    def send_request(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Send a prompt to a specified model and return a standardized LLMResponse.
        Handles retries, timing, and token estimation.
        """
        model_cfg = self.registry.get(model_name)
        if model_cfg is None:
            return self._error_response(model_name, f"Unknown model: {model_name}")

        payload = {
            "model": model_cfg.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        last_error = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                start = time.time()
                response = httpx.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )
                latency_ms = (time.time() - start) * 1000
                response.raise_for_status()
                data = response.json()
                return self._build_response(model_name, model_cfg, data, latency_ms, prompt)

            except httpx.ConnectError as e:
                last_error = f"Ollama not reachable at {self.base_url}: {e}"
                logger.warning(f"Attempt {attempt}/{self.retry_attempts} failed: {last_error}")
            except httpx.TimeoutException as e:
                last_error = f"Request timed out after {self.timeout}s: {e}"
                logger.warning(f"Attempt {attempt}/{self.retry_attempts} timeout: {last_error}")
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text}"
                logger.error(f"HTTP error on attempt {attempt}: {last_error}")
                break  # Don't retry on HTTP errors (model not found, etc.)
            except Exception as e:
                last_error = f"Unexpected error: {e}"
                logger.error(f"Unexpected error on attempt {attempt}: {last_error}")

            if attempt < self.retry_attempts:
                time.sleep(self.retry_delay)

        return self._error_response(model_name, last_error or "Unknown error")

    def _build_response(
        self,
        model_name: str,
        model_cfg: ModelConfig,
        data: dict,
        latency_ms: float,
        prompt: str,
    ) -> LLMResponse:
        content = data.get("response", "")

        # Ollama returns eval_count (completion tokens) and prompt_eval_count
        completion_tokens = data.get("eval_count", self._estimate_tokens(content))
        prompt_tokens = data.get("prompt_eval_count", self._estimate_tokens(prompt))
        total_tokens = prompt_tokens + completion_tokens
        cost = model_cfg.estimated_cost(total_tokens)

        return LLMResponse(
            content=content,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            cost=cost,
            raw_response=data,
        )

    def _error_response(self, model_name: str, error: str) -> LLMResponse:
        return LLMResponse(
            content="",
            model_name=model_name,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency_ms=0.0,
            cost=0.0,
            error=error,
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return max(1, len(text) // 4)

    def check_model_available(self, ollama_model: str) -> bool:
        """Check if a specific model is available in Ollama."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=10)
            if resp.status_code == 200:
                models = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
                return ollama_model in models
        except Exception:
            pass
        return False

    def list_available_models(self) -> list[str]:
        """Return list of model names available in Ollama."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=10)
            if resp.status_code == 200:
                return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            pass
        return []
