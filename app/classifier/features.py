"""
app/classifier/features.py
Feature extraction for complexity classification.
Extracts linguistic and structural features from prompts.
"""

import re
from typing import List


# Keywords that signal high complexity
COMPLEX_KEYWORDS = {
    "design", "architect", "implement", "distributed", "scalable", "system",
    "fault-tolerant", "concurrent", "trade-offs", "real-time", "billion",
    "million", "global", "enterprise", "pipeline", "consensus", "sharding",
    "migration", "platform", "infrastructure", "compliance", "multi-region",
    "exactly-once", "zero-downtime", "hyperparameter", "federated", "semantic",
    "orchestration", "canary", "observability", "anomaly", "collaborative",
}

MEDIUM_KEYWORDS = {
    "explain", "write", "difference", "how does", "function", "class",
    "algorithm", "pattern", "query", "script", "compare", "analyze",
    "describe", "what are", "implement", "example", "between", "work",
    "build", "create", "develop",
}

SIMPLE_KEYWORDS = {
    "what is", "who is", "define", "translate", "convert", "list",
    "name", "when", "where", "how many", "spell", "abbreviation", "capital",
    "formula", "calculate", "year",
}

QUESTION_WORDS = {"what", "who", "when", "where", "how", "why", "which"}


def extract_features(prompt: str) -> List[float]:
    """
    Extract numerical features from a prompt for classification.

    Feature vector:
        [0]  word count
        [1]  char count (normalized)
        [2]  sentence count
        [3]  avg word length
        [4]  question word ratio
        [5]  complex keyword count
        [6]  medium keyword count
        [7]  simple keyword count
        [8]  technical term density
        [9]  has code request indicator
        [10] has design/architect indicator
        [11] has numbers > 1000 indicator
        [12] has comparison indicator
        [13] punctuation density
        [14] starts with question word
    """
    text = prompt.strip()
    lower = text.lower()
    words = re.findall(r"\b\w+\b", lower)
    word_count = len(words)
    if word_count == 0:
        return [0.0] * 15

    char_count = len(text) / 500.0  # normalized
    sentences = re.split(r"[.!?]", text)
    sentence_count = max(1, len([s for s in sentences if s.strip()]))
    avg_word_len = sum(len(w) for w in words) / word_count

    # Question word ratio
    q_words = sum(1 for w in words if w in QUESTION_WORDS)
    q_ratio = q_words / word_count

    # Keyword counts
    complex_count = sum(1 for kw in COMPLEX_KEYWORDS if kw in lower)
    medium_count = sum(1 for kw in MEDIUM_KEYWORDS if kw in lower)
    simple_count = sum(1 for kw in SIMPLE_KEYWORDS if kw in lower)

    # Technical density (long words often = technical)
    tech_words = sum(1 for w in words if len(w) > 7)
    tech_density = tech_words / word_count

    # Code request: "write a function", "implement", "code"
    has_code = float(
        any(t in lower for t in ["write a function", "implement", "script", "code", "class", "query", "algorithm"])
    )

    # Design/architect
    has_design = float(
        any(t in lower for t in ["design", "architect", "build a system", "platform"])
    )

    # Large numbers (scale indicators)
    numbers = re.findall(r"\b\d+[kmb]?\b", lower)
    has_large_numbers = float(
        any(
            int(re.sub(r"[kmb]", "", n)) > 1000
            for n in numbers
            if re.sub(r"[kmb]", "", n).isdigit()
        )
    )

    # Comparison
    has_comparison = float(
        any(t in lower for t in ["difference between", "compare", "vs", "versus", "trade-off"])
    )

    # Punctuation density
    punct_count = len(re.findall(r"[,;:()\[\]{}]", text))
    punct_density = punct_count / max(1, word_count)

    # Starts with question word
    first_word = words[0] if words else ""
    starts_question = float(first_word in QUESTION_WORDS)

    return [
        word_count / 50.0,       # normalized word count
        char_count,
        sentence_count / 5.0,    # normalized
        avg_word_len / 10.0,     # normalized
        q_ratio,
        complex_count / 5.0,     # normalized
        medium_count / 5.0,
        simple_count / 5.0,
        tech_density,
        has_code,
        has_design,
        has_large_numbers,
        has_comparison,
        punct_density,
        starts_question,
    ]
