"""
app/models/registry.py
ModelRegistry: loads and provides access to all configured models.
"""

import yaml
from pathlib import Path
from typing import Dict, Optional
from app.models.schemas import ModelConfig, ModelTier


class ModelRegistry:
    """Central registry for all configured LLM models."""

    def __init__(self, config_path: str = "configs/models.yaml"):
        self._models: Dict[str, ModelConfig] = {}
        self._raw_config: dict = {}
        self._load(config_path)

    def _load(self, config_path: str) -> None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")

        with open(path) as f:
            self._raw_config = yaml.safe_load(f)

        for name, cfg in self._raw_config.get("models", {}).items():
            self._models[name] = ModelConfig(
                name=cfg["name"],
                tier=ModelTier(cfg["tier"]),
                cost_per_token=cfg["cost_per_token"],
                max_tokens=cfg["max_tokens"],
                context_window=cfg["context_window"],
                description=cfg["description"],
                ollama_model=cfg["ollama_model"],
            )

    def get(self, name: str) -> Optional[ModelConfig]:
        return self._models.get(name)

    def get_by_tier(self, tier: ModelTier) -> Optional[ModelConfig]:
        for model in self._models.values():
            if model.tier == tier:
                return model
        return None

    def all(self) -> Dict[str, ModelConfig]:
        return dict(self._models)

    def names(self) -> list[str]:
        return list(self._models.keys())

    @property
    def ollama_config(self) -> dict:
        return self._raw_config.get("ollama", {})

    @property
    def baseline_model(self) -> str:
        return self._raw_config.get("baseline_model", "llama3")

    @property
    def baseline_cost_per_1k(self) -> float:
        return self._raw_config.get("baseline_cost_per_1k_tokens", 0.002)
