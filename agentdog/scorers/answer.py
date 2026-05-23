from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .base import ScoreResult, Scorer

if TYPE_CHECKING:
    from agentdog.trace import AgentTrace


class ContainsAnswer(Scorer):
    """Pass if the output contains all required substrings."""

    def __init__(self, required: list[str], case_sensitive: bool = False):
        self.required = required
        self.case_sensitive = case_sensitive

    def score(self, trace: "AgentTrace") -> ScoreResult:
        output = trace.output if self.case_sensitive else trace.output.lower()
        missing = []
        for phrase in self.required:
            needle = phrase if self.case_sensitive else phrase.lower()
            if needle not in output:
                missing.append(phrase)
        passed = len(missing) == 0
        return ScoreResult(
            passed=passed,
            score=1.0 - len(missing) / len(self.required),
            reason="" if passed else f"Missing from output: {missing}",
        )


class ExactAnswer(Scorer):
    """Pass if the output exactly equals the expected string (after optional strip)."""

    def __init__(self, expected: str, strip: bool = True):
        self.expected = expected
        self.strip = strip

    def score(self, trace: "AgentTrace") -> ScoreResult:
        actual = trace.output.strip() if self.strip else trace.output
        expected = self.expected.strip() if self.strip else self.expected
        passed = actual == expected
        return ScoreResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reason="" if passed else f"Expected {expected!r}, got {actual!r}",
        )


class RegexAnswer(Scorer):
    """Pass if the output matches the given regex pattern."""

    def __init__(self, pattern: str, flags: int = re.IGNORECASE):
        self.pattern = pattern
        self.flags = flags
        self._re = re.compile(pattern, flags)

    def score(self, trace: "AgentTrace") -> ScoreResult:
        match = self._re.search(trace.output)
        passed = match is not None
        return ScoreResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reason="" if passed else f"Pattern {self.pattern!r} not found in output",
        )


class ForbiddenContent(Scorer):
    """Pass if the output contains none of the forbidden substrings."""

    def __init__(self, forbidden: list[str], case_sensitive: bool = False):
        self.forbidden = forbidden
        self.case_sensitive = case_sensitive

    def score(self, trace: "AgentTrace") -> ScoreResult:
        output = trace.output if self.case_sensitive else trace.output.lower()
        found = []
        for phrase in self.forbidden:
            needle = phrase if self.case_sensitive else phrase.lower()
            if needle in output:
                found.append(phrase)
        passed = len(found) == 0
        return ScoreResult(
            passed=passed,
            score=1.0 - len(found) / len(self.forbidden),
            reason="" if passed else f"Forbidden content found: {found}",
        )


class AnswerNotEmpty(Scorer):
    """Pass if the output is non-empty after stripping whitespace."""

    def __init__(self, min_chars: int = 1):
        self.min_chars = min_chars

    def score(self, trace: "AgentTrace") -> ScoreResult:
        length = len(trace.output.strip())
        passed = length >= self.min_chars
        return ScoreResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reason="" if passed else f"Output too short: {length} chars (min {self.min_chars})",
        )
