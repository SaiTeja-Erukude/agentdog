from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .base import ScoreResult, Scorer

if TYPE_CHECKING:
    from agentdog.trace import AgentTrace


MissingDataPolicy = Literal["skip", "fail", "error", "pass"]


def _validate_missing(missing: MissingDataPolicy) -> None:
    if missing not in ("skip", "fail", "error", "pass"):
        raise ValueError(f"Unknown missing-data policy: {missing!r}")


def _missing_result(metric: str, policy: MissingDataPolicy) -> ScoreResult:
    return ScoreResult(
        passed=policy == "pass",
        score=1.0 if policy == "pass" else 0.0,
        status=policy,
        reason=f"{metric} not available in trace (missing-data policy: {policy})",
        details={"missing_data": True, "policy": policy},
    )


class UnderTokenLimit(Scorer):
    """Pass if total token usage is at or below the limit."""

    def __init__(self, max_tokens: int, *, missing: MissingDataPolicy = "skip"):
        _validate_missing(missing)
        self.max_tokens = max_tokens
        self.missing = missing

    def score(self, trace: "AgentTrace") -> ScoreResult:
        if trace.total_tokens is None:
            return _missing_result("Token count", self.missing)
        passed = trace.total_tokens <= self.max_tokens
        ratio = trace.total_tokens / self.max_tokens
        return ScoreResult(
            passed=passed,
            score=max(0.0, 1.0 - (ratio - 1.0)) if not passed else 1.0,
            reason="" if passed else f"Token usage {trace.total_tokens} exceeds limit {self.max_tokens}",
            details={"actual": trace.total_tokens, "max": self.max_tokens},
        )


class UnderCostLimit(Scorer):
    """Pass if total cost is at or below the USD limit."""

    def __init__(self, max_cost_usd: float, *, missing: MissingDataPolicy = "skip"):
        _validate_missing(missing)
        self.max_cost_usd = max_cost_usd
        self.missing = missing

    def score(self, trace: "AgentTrace") -> ScoreResult:
        if trace.total_cost_usd is None:
            return _missing_result("Cost", self.missing)
        passed = trace.total_cost_usd <= self.max_cost_usd
        ratio = trace.total_cost_usd / self.max_cost_usd if self.max_cost_usd else float("inf")
        return ScoreResult(
            passed=passed,
            score=max(0.0, 1.0 - (ratio - 1.0)) if not passed else 1.0,
            reason="" if passed else (
                f"Cost ${trace.total_cost_usd:.4f} exceeds limit ${self.max_cost_usd:.4f}"
            ),
            details={"actual_usd": trace.total_cost_usd, "max_usd": self.max_cost_usd},
        )


class UnderLatencyLimit(Scorer):
    """Pass if total latency is at or below the limit in milliseconds."""

    def __init__(self, max_latency_ms: float, *, missing: MissingDataPolicy = "skip"):
        _validate_missing(missing)
        self.max_latency_ms = max_latency_ms
        self.missing = missing

    def score(self, trace: "AgentTrace") -> ScoreResult:
        if trace.total_latency_ms is None:
            return _missing_result("Latency", self.missing)
        passed = trace.total_latency_ms <= self.max_latency_ms
        ratio = trace.total_latency_ms / self.max_latency_ms
        return ScoreResult(
            passed=passed,
            score=max(0.0, 1.0 - (ratio - 1.0)) if not passed else 1.0,
            reason="" if passed else (
                f"Latency {trace.total_latency_ms:.0f}ms exceeds limit {self.max_latency_ms:.0f}ms"
            ),
            details={"actual_ms": trace.total_latency_ms, "max_ms": self.max_latency_ms},
        )


class MaxRetries(Scorer):
    """Pass if the number of retries is at or below the limit."""

    def __init__(self, max_retries: int):
        self.max_retries = max_retries

    def score(self, trace: "AgentTrace") -> ScoreResult:
        passed = trace.num_retries <= self.max_retries
        return ScoreResult(
            passed=passed,
            score=1.0 if passed else max(0.0, 1.0 - (trace.num_retries - self.max_retries) / (self.max_retries + 1)),
            reason="" if passed else f"Retries {trace.num_retries} exceeds max {self.max_retries}",
            details={"actual": trace.num_retries, "max": self.max_retries},
        )
