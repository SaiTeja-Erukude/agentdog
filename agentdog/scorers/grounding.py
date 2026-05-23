from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .base import ScoreResult, Scorer

if TYPE_CHECKING:
    from agentdog.trace import AgentTrace


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


class GroundedInContext(Scorer):
    """
    Pass if the answer's content overlaps sufficiently with the retrieved context.

    Uses word-level Jaccard overlap between the answer and the union of all
    retrieved context chunks. For semantic grounding, use LLMJudge instead.
    """

    def __init__(self, threshold: float = 0.25):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold

    def score(self, trace: "AgentTrace") -> ScoreResult:
        if not trace.retrieved_context:
            return ScoreResult(
                passed=False,
                score=0.0,
                reason="No retrieved context in trace",
            )
        context_words = _tokenize(" ".join(trace.retrieved_context))
        answer_words = _tokenize(trace.output)
        if not answer_words:
            return ScoreResult(passed=False, score=0.0, reason="Empty answer")

        overlap = len(answer_words & context_words) / len(answer_words)
        passed = overlap >= self.threshold
        return ScoreResult(
            passed=passed,
            score=overlap,
            reason="" if passed else f"Answer-context word overlap {overlap:.2%} < threshold {self.threshold:.2%}",
            details={"overlap": overlap, "threshold": self.threshold},
        )


class CitedSource(Scorer):
    """
    Pass if the answer contains at least one citation marker.

    Looks for common patterns: [1], [Source], (Source: ...), footnote markers,
    or any of the user-supplied custom patterns.
    """

    _DEFAULT_PATTERNS = [
        r"\[\d+\]",              # [1], [23]
        r"\[source[^\]]*\]",     # [source], [Source: doc]
        r"\(source[^)]*\)",      # (source: ...)
        r"according to",
        r"as stated in",
        r"per the",
        r"based on",
    ]

    def __init__(self, custom_patterns: list[str] | None = None):
        patterns = (custom_patterns or []) + self._DEFAULT_PATTERNS
        self._re = re.compile("|".join(patterns), re.IGNORECASE)

    def score(self, trace: "AgentTrace") -> ScoreResult:
        match = self._re.search(trace.output)
        passed = match is not None
        return ScoreResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reason="" if passed else "No citation markers found in answer",
        )


class NoContextHallucination(Scorer):
    """
    Pass if every sentence in the answer can be loosely traced back to
    retrieved context (word-overlap heuristic per sentence).

    This is a lightweight proxy. For rigorous hallucination detection use LLMJudge.
    """

    def __init__(self, min_sentence_overlap: float = 0.2):
        self.min_sentence_overlap = min_sentence_overlap

    def score(self, trace: "AgentTrace") -> ScoreResult:
        if not trace.retrieved_context:
            return ScoreResult(
                passed=False,
                score=0.0,
                reason="No retrieved context in trace",
            )
        context_words = _tokenize(" ".join(trace.retrieved_context))
        sentences = [s.strip() for s in re.split(r"[.!?]+", trace.output) if s.strip()]
        if not sentences:
            return ScoreResult(passed=True, score=1.0)

        grounded, ungrounded = [], []
        for s in sentences:
            words = _tokenize(s)
            if not words:
                continue
            overlap = len(words & context_words) / len(words)
            if overlap >= self.min_sentence_overlap:
                grounded.append(s)
            else:
                ungrounded.append(s)

        total = len(grounded) + len(ungrounded)
        score = len(grounded) / total if total else 1.0
        passed = len(ungrounded) == 0
        return ScoreResult(
            passed=passed,
            score=score,
            reason="" if passed else f"{len(ungrounded)} sentence(s) not grounded: {ungrounded[:2]}",
            details={"grounded": len(grounded), "ungrounded": len(ungrounded)},
        )
