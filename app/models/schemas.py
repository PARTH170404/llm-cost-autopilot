"""
app/models/schemas.py
Pydantic schemas and dataclasses for the LLM Cost Autopilot system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel


class ComplexityTier(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class ModelTier(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


@dataclass
class ModelConfig:
    """Configuration for a single LLM model."""
    name: str
    tier: ModelTier
    cost_per_token: float
    max_tokens: int
    context_window: int
    description: str
    ollama_model: str

    def estimated_cost(self, tokens: int) -> float:
        return self.cost_per_token * tokens


@dataclass
class LLMResponse:
    """Standardized response object from any LLM."""
    content: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    raw_response: Optional[dict] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.content)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "model_name": self.model_name,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "cost": self.cost,
            "timestamp": self.timestamp.isoformat(),
            "error": self.error,
        }


# ── Pydantic API schemas ──────────────────────────────────────────────────────

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    force_model: Optional[str] = None  # Override routing

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "What is the capital of France?",
                "max_tokens": 256,
                "temperature": 0.7,
            }
        }


class CompletionResponse(BaseModel):
    content: str
    model_used: str
    complexity_tier: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost: float
    estimated_savings: float
    request_id: str


class ModelInfo(BaseModel):
    name: str
    tier: str
    description: str
    available: bool


class StatsResponse(BaseModel):
    total_requests: int
    total_tokens: int
    total_cost: float
    total_savings: float
    avg_latency_ms: float
    model_distribution: dict
    escalation_rate: float
