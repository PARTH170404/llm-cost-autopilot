"""
app/services/autopilot.py
AutopilotService: orchestrates routing → inference → evaluation → logging.
This is the single entry point for all completion requests.
"""

import logging
import asyncio
from typing import Optional

from app.models.schemas import ComplexityTier, LLMResponse
from app.models.registry import ModelRegistry
from app.classifier.complexity_classifier import ComplexityClassifier
from app.router.routing_engine import RoutingEngine
from app.services.ollama_client import OllamaClient
from app.evaluator.quality_evaluator import QualityEvaluator
from app.db.database import Database
from app.utils.config import get_setting

logger = logging.getLogger(__name__)

MAX_ESCALATIONS = int(get_setting("routing.max_escalation_attempts", 2))
ENABLE_ESCALATION = bool(get_setting("routing.enable_escalation", True))
ASYNC_EVAL = bool(get_setting("evaluator.async_eval", True))


class AutopilotResult:
    def __init__(
        self,
        request_id: str,
        response: LLMResponse,
        complexity_tier: ComplexityTier,
        confidence: float,
        estimated_savings: float,
        escalated: bool = False,
        quality_score: float | None = None,
    ):
        self.request_id = request_id
        self.response = response
        self.complexity_tier = complexity_tier
        self.confidence = confidence
        self.estimated_savings = estimated_savings
        self.escalated = escalated
        self.quality_score = quality_score


class AutopilotService:
    """
    End-to-end orchestration:
    1. Classify prompt complexity
    2. Route to cheapest capable model
    3. Call Ollama
    4. Evaluate quality (async)
    5. Escalate if quality is poor
    6. Log everything to SQLite
    """

    def __init__(self):
        self.registry = ModelRegistry("configs/models.yaml")
        self.classifier = ComplexityClassifier()
        self.router = RoutingEngine(self.registry, self.classifier)
        self.client = OllamaClient(self.registry)
        self.evaluator = QualityEvaluator()
        self.db = Database()
        self._baseline_cost_per_token = self.registry.baseline_cost_per_1k / 1000.0

    # ── Public API ────────────────────────────────────────────────────────────

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        force_model: Optional[str] = None,
    ) -> AutopilotResult:
        """Full autopilot pipeline — async entry point."""

        # 1. Route
        model_cfg, tier, confidence = self.router.route(prompt, force_model)

        # 2. Inference (run in thread to not block event loop)
        loop = asyncio.get_event_loop()
        response: LLMResponse = await loop.run_in_executor(
            None,
            lambda: self.client.send_request(
                model_name=model_cfg.name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            ),
        )

        # 3. Evaluate quality
        quality_score = None
        escalated = False

        if ASYNC_EVAL and response.success:
            quality_score = await self.evaluator.score_async(prompt, response)

            # 4. Escalate if quality is poor
            if ENABLE_ESCALATION and self.evaluator.should_escalate(quality_score):
                escalated_response, escalated_model = await self._try_escalate(
                    prompt, tier, max_tokens, temperature
                )
                if escalated_response and escalated_response.success:
                    new_score = await self.evaluator.score_async(prompt, escalated_response)
                    if new_score > quality_score:
                        # Log escalation event
                        escalated = True
                        old_model = response.model_name
                        response = escalated_response
                        tier = ComplexityTier(escalated_model.tier.value)
                        quality_score = new_score
                        logger.info(f"Escalation improved score: {quality_score:.2f}")
                        # We'll log escalation after we have the request_id

        # 5. Compute savings
        baseline_cost = self._baseline_cost_per_token * response.total_tokens
        estimated_savings = max(0.0, baseline_cost - response.cost)

        # 6. Log to DB
        request_id = self.db.log_request(
            prompt=prompt,
            complexity_tier=tier.value,
            confidence=confidence,
            model_used=response.model_name,
            content=response.content,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            latency_ms=response.latency_ms,
            cost=response.cost,
            estimated_savings=estimated_savings,
            quality_score=quality_score,
            escalated=escalated,
            forced_model=force_model,
            error=response.error,
        )

        return AutopilotResult(
            request_id=request_id,
            response=response,
            complexity_tier=tier,
            confidence=confidence,
            estimated_savings=estimated_savings,
            escalated=escalated,
            quality_score=quality_score,
        )

    async def _try_escalate(self, prompt, current_tier, max_tokens, temperature):
        """Attempt escalation to a higher-tier model."""
        next_model = self.router.escalate(current_tier)
        if next_model is None:
            return None, None

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.send_request(
                model_name=next_model.name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        return response, next_model

    def get_stats(self) -> dict:
        return self.db.get_stats()

    def get_recent_requests(self, limit: int = 50) -> list[dict]:
        return self.db.get_recent_requests(limit)

    def get_available_models(self) -> list[dict]:
        available_ollama = self.client.list_available_models()
        result = []
        for name, cfg in self.registry.all().items():
            result.append({
                "name": name,
                "tier": cfg.tier.value,
                "description": cfg.description,
                "available": any(cfg.ollama_model in m for m in available_ollama),
            })
        return result
