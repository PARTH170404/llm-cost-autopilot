"""
app/router/routing_engine.py
Routes prompts to the appropriate model based on complexity tier.
Integrates with ComplexityClassifier and ModelRegistry.
"""

import logging
from app.models.schemas import ComplexityTier, ModelConfig
from app.models.registry import ModelRegistry
from app.classifier.complexity_classifier import ComplexityClassifier

logger = logging.getLogger(__name__)

# Tier → model name mapping (mirrors configs/models.yaml tiers)
TIER_MODEL_MAP = {
    ComplexityTier.SIMPLE:  "phi3",
    ComplexityTier.MEDIUM:  "mistral",
    ComplexityTier.COMPLEX: "llama3",
}


class RoutingEngine:
    """
    Routes a prompt to the cheapest model capable of handling its complexity.
    Supports forced model override and escalation.
    """

    def __init__(self, registry: ModelRegistry, classifier: ComplexityClassifier):
        self.registry = registry
        self.classifier = classifier

    def route(self, prompt: str, force_model: str | None = None) -> tuple[ModelConfig, ComplexityTier, float]:
        """
        Returns (ModelConfig, ComplexityTier, confidence).
        If force_model is set, bypasses classification.
        """
        if force_model:
            model_cfg = self.registry.get(force_model)
            if model_cfg is None:
                logger.warning(f"Forced model '{force_model}' not found. Using fallback.")
                model_cfg = self._fallback()
            tier = ComplexityTier(model_cfg.tier.value)
            logger.info(f"Forced routing → {model_cfg.name} (tier={tier})")
            return model_cfg, tier, 1.0

        tier, confidence = self.classifier.classify_with_confidence(prompt)
        model_name = TIER_MODEL_MAP.get(tier, "llama3")
        model_cfg = self.registry.get(model_name)

        if model_cfg is None:
            logger.warning(f"Mapped model '{model_name}' not in registry. Using fallback.")
            model_cfg = self._fallback()

        logger.info(
            f"Routed prompt (len={len(prompt)}) → {model_cfg.name} "
            f"[tier={tier}, confidence={confidence:.2f}]"
        )
        return model_cfg, tier, confidence

    def escalate(self, current_tier: ComplexityTier) -> ModelConfig | None:
        """
        Returns the next higher-tier model, or None if already at the top.
        """
        escalation_path = {
            ComplexityTier.SIMPLE:  ComplexityTier.MEDIUM,
            ComplexityTier.MEDIUM:  ComplexityTier.COMPLEX,
            ComplexityTier.COMPLEX: None,
        }
        next_tier = escalation_path.get(current_tier)
        if next_tier is None:
            logger.info("Already at highest tier — cannot escalate further.")
            return None

        model_name = TIER_MODEL_MAP.get(next_tier)
        model_cfg = self.registry.get(model_name)
        logger.info(f"Escalating from {current_tier} → {next_tier} ({model_name})")
        return model_cfg

    def _fallback(self) -> ModelConfig:
        fallback_name = self.registry.baseline_model
        cfg = self.registry.get(fallback_name)
        if cfg is None:
            # Last resort: first model in registry
            cfg = list(self.registry.all().values())[0]
        return cfg
