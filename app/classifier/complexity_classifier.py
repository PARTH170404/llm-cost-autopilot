"""
app/classifier/complexity_classifier.py
Loads trained classifier model and classifies prompt complexity.
"""

import pickle
import logging
from pathlib import Path
from app.classifier.features import extract_features
from app.models.schemas import ComplexityTier

logger = logging.getLogger(__name__)


class ComplexityClassifier:
    """
    Classifies prompts into simple / medium / complex tiers.
    Uses pre-trained GradientBoosting classifier with hand-crafted features.
    """

    def __init__(
        self,
        model_path: str = "data/classifier.pkl",
        label_encoder_path: str = "data/label_encoder.pkl",
    ):
        self.model_path = model_path
        self.label_encoder_path = label_encoder_path
        self.model = None
        self.label_encoder = None
        self._load()

    def _load(self) -> None:
        mp = Path(self.model_path)
        lp = Path(self.label_encoder_path)

        if not mp.exists():
            raise FileNotFoundError(
                f"Classifier model not found at {self.model_path}. "
                "Run: python scripts/train_classifier.py"
            )
        if not lp.exists():
            raise FileNotFoundError(
                f"Label encoder not found at {self.label_encoder_path}. "
                "Run: python scripts/train_classifier.py"
            )

        with open(mp, "rb") as f:
            self.model = pickle.load(f)
        with open(lp, "rb") as f:
            self.label_encoder = pickle.load(f)

        logger.info(f"Classifier loaded from {self.model_path}")

    def classify(self, prompt: str) -> ComplexityTier:
        """Classify a prompt and return its ComplexityTier."""
        features = [extract_features(prompt)]
        pred_idx = self.model.predict(features)[0]
        label = self.label_encoder.inverse_transform([pred_idx])[0]
        return ComplexityTier(label)

    def classify_with_confidence(self, prompt: str) -> tuple[ComplexityTier, float]:
        """Return tier + confidence probability."""
        features = [extract_features(prompt)]
        pred_idx = self.model.predict(features)[0]
        label = self.label_encoder.inverse_transform([pred_idx])[0]

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features)[0]
            confidence = float(probs[pred_idx])
        else:
            confidence = 1.0

        return ComplexityTier(label), confidence

    def is_loaded(self) -> bool:
        return self.model is not None and self.label_encoder is not None
