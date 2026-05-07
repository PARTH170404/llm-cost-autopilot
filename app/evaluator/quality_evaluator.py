"""
app/evaluator/quality_evaluator.py
Rule-based quality evaluator with async escalation support.
Scores responses on length, coherence, and completeness heuristics.
"""

import re
import asyncio
import logging
from app.models.schemas import LLMResponse
from app.utils.config import get_setting

logger = logging.getLogger(__name__)

MIN_QUALITY = float(get_setting("evaluator.min_quality_score", 0.6))
ESCALATION_THRESHOLD = float(get_setting("evaluator.escalation_threshold", 0.5))


class QualityEvaluator:
    """
    Scores LLM responses using rule-based heuristics.
    Returns a float [0, 1] where 1 is highest quality.
    """

    def score(self, prompt: str, response: LLMResponse) -> float:
        if not response.success:
            return 0.0

        content = response.content.strip()
        if not content:
            return 0.0

        scores = [
            self._length_score(prompt, content),
            self._completeness_score(content),
            self._coherence_score(content),
            self._relevance_score(prompt, content),
        ]
        final = sum(scores) / len(scores)
        logger.debug(f"Quality scores: {scores} → final={final:.2f}")
        return round(final, 3)

    # ── Individual scorers ────────────────────────────────────────────────────

    def _length_score(self, prompt: str, content: str) -> float:
        """Penalise too-short or truncated responses."""
        word_count = len(content.split())
        prompt_words = len(prompt.split())

        # Expect at least ~1 word out per word in, up to 20 minimum
        min_words = max(20, prompt_words)

        if word_count < 5:
            return 0.1
        elif word_count < min_words * 0.3:
            return 0.4
        elif word_count < min_words:
            return 0.7
        else:
            # Bonus for thorough responses (cap at 1.0)
            return min(1.0, 0.8 + (word_count / (min_words * 3)) * 0.2)

    def _completeness_score(self, content: str) -> float:
        """Check that the response doesn't trail off unexpectedly."""
        # Penalise if ending looks truncated
        stripped = content.strip()
        if not stripped:
            return 0.0
        last_char = stripped[-1]
        if last_char in ".!?":
            return 1.0
        elif last_char in ",;:":
            return 0.5
        else:
            return 0.7  # ends mid-sentence — neutral

    def _coherence_score(self, content: str) -> float:
        """Basic coherence: check for repeated phrases, excessive filler."""
        words = content.lower().split()
        if not words:
            return 0.0

        # Detect excessive repetition
        word_freq: dict[str, int] = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
        max_freq = max(word_freq.values())
        repetition_ratio = max_freq / len(words)

        if repetition_ratio > 0.3:
            return 0.3  # Heavy repetition
        elif repetition_ratio > 0.15:
            return 0.7
        else:
            return 1.0

    def _relevance_score(self, prompt: str, content: str) -> float:
        """Check content shares vocabulary with prompt (basic relevance)."""
        prompt_tokens = set(re.findall(r"\b\w{4,}\b", prompt.lower()))
        content_tokens = set(re.findall(r"\b\w{4,}\b", content.lower()))
        if not prompt_tokens:
            return 1.0
        overlap = prompt_tokens & content_tokens
        ratio = len(overlap) / len(prompt_tokens)
        # Even 20% overlap is fine for generative responses
        return min(1.0, ratio * 2.5)

    # ── Escalation decision ───────────────────────────────────────────────────

    def should_escalate(self, score: float) -> bool:
        return score < ESCALATION_THRESHOLD

    # ── Async wrapper ─────────────────────────────────────────────────────────

    async def score_async(self, prompt: str, response: LLMResponse) -> float:
        """Run scoring in a thread to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.score, prompt, response)
