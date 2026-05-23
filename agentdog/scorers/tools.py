from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .base import ScoreResult, Scorer

if TYPE_CHECKING:
    from agentdog.trace import AgentTrace


class UsedTools(Scorer):
    """Pass if all required tools were called at least once."""

    def __init__(self, required: list[str]):
        self.required = required

    def score(self, trace: "AgentTrace") -> ScoreResult:
        called = set(trace.tool_names)
        missing = [t for t in self.required if t not in called]
        passed = len(missing) == 0
        return ScoreResult(
            passed=passed,
            score=1.0 - len(missing) / len(self.required),
            reason="" if passed else f"Required tools not called: {missing}",
            details={"called": list(called), "required": self.required},
        )


class AvoidedTools(Scorer):
    """Pass if none of the forbidden tools were called."""

    def __init__(self, forbidden: list[str]):
        self.forbidden = forbidden

    def score(self, trace: "AgentTrace") -> ScoreResult:
        called = set(trace.tool_names)
        violations = [t for t in self.forbidden if t in called]
        passed = len(violations) == 0
        return ScoreResult(
            passed=passed,
            score=1.0 - len(violations) / len(self.forbidden),
            reason="" if passed else f"Forbidden tools were called: {violations}",
            details={"called": list(called), "forbidden": self.forbidden},
        )


class ToolCallOrder(Scorer):
    """Pass if the required tools were called in the specified order (as a subsequence)."""

    def __init__(self, ordered: list[str]):
        self.ordered = ordered

    def score(self, trace: "AgentTrace") -> ScoreResult:
        names = trace.tool_names
        it = iter(names)
        matched = 0
        for step in self.ordered:
            if any(name == step for name in it):
                matched += 1
        passed = matched == len(self.ordered)
        return ScoreResult(
            passed=passed,
            score=matched / len(self.ordered) if self.ordered else 1.0,
            reason="" if passed else f"Expected order {self.ordered}, got {names}",
        )


class MaxToolCalls(Scorer):
    """Pass if total tool calls do not exceed the limit."""

    def __init__(self, max_calls: int):
        self.max_calls = max_calls

    def score(self, trace: "AgentTrace") -> ScoreResult:
        n = len(trace.tool_calls)
        passed = n <= self.max_calls
        return ScoreResult(
            passed=passed,
            score=min(1.0, self.max_calls / n) if n > 0 else 1.0,
            reason="" if passed else f"Too many tool calls: {n} (max {self.max_calls})",
            details={"actual": n, "max": self.max_calls},
        )


class ToolArgContains(Scorer):
    """
    Pass if a specific tool was called with an argument whose string
    representation contains the expected value.
    """

    def __init__(self, tool: str, arg: str, expected: Any, case_sensitive: bool = False):
        self.tool = tool
        self.arg = arg
        self.expected = expected
        self.case_sensitive = case_sensitive

    def score(self, trace: "AgentTrace") -> ScoreResult:
        calls = trace.get_tool_calls(self.tool)
        if not calls:
            return ScoreResult(
                passed=False,
                score=0.0,
                reason=f"Tool {self.tool!r} was never called",
            )
        needle = str(self.expected)
        if not self.case_sensitive:
            needle = needle.lower()
        for call in calls:
            val = str(call.arguments.get(self.arg, ""))
            if not self.case_sensitive:
                val = val.lower()
            if needle in val:
                return ScoreResult(passed=True, score=1.0)
        return ScoreResult(
            passed=False,
            score=0.0,
            reason=(
                f"Tool {self.tool!r} argument {self.arg!r} never contained "
                f"{self.expected!r}. Got: "
                f"{[c.arguments.get(self.arg) for c in calls]}"
            ),
        )


class ToolArgEquals(Scorer):
    """Pass if a specific tool was called with an argument equal to the expected value."""

    def __init__(self, tool: str, arg: str, expected: Any):
        self.tool = tool
        self.arg = arg
        self.expected = expected

    def score(self, trace: "AgentTrace") -> ScoreResult:
        calls = trace.get_tool_calls(self.tool)
        if not calls:
            return ScoreResult(
                passed=False,
                score=0.0,
                reason=f"Tool {self.tool!r} was never called",
            )
        for call in calls:
            if call.arguments.get(self.arg) == self.expected:
                return ScoreResult(passed=True, score=1.0)
        return ScoreResult(
            passed=False,
            score=0.0,
            reason=(
                f"Tool {self.tool!r} argument {self.arg!r} never equaled "
                f"{self.expected!r}. Got: "
                f"{[c.arguments.get(self.arg) for c in calls]}"
            ),
        )
