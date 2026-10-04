"""Assertion helpers for pytest and other Python test runners.

No pytest dependency is required. Assertions remain active under Python -O.
"""
from __future__ import annotations

from collections.abc import Sequence

from .case import EvalRun, TestCase
from .report import CaseResult
from .runner import run
from .scorers.base import Scorer
from .trace import AgentTrace


def assert_evaluates(
    trace: AgentTrace,
    scorers: Sequence[Scorer],
    *,
    name: str = "agent evaluation",
    allow_skips: bool = False,
) -> CaseResult:
    """Evaluate a trace once and raise AssertionError with scorer explanations.

    Skipped checks fail the assertion by default so missing evidence is visible
    in tests. Set allow_skips=True to accept them. Return results on success.
    """
    __tracebackhide__ = True  # pytest hides this frame in failure output
    if not scorers:
        raise ValueError("assert_evaluates requires at least one scorer")
    case = TestCase(name=name, scorers=list(scorers))
    result = run([EvalRun(case=case, trace=trace)]).case_results[0]
    problems = [
        sr for sr in result.scorer_results
        if sr.result.status in ("fail", "error")
        or (sr.result.status == "skip" and not allow_skips)
    ]
    if problems:
        lines = [f"{name}: {len(problems)} scorer check(s) did not pass"]
        for sr in problems:
            lines.append(
                f"  [{sr.result.status.upper()}] {sr.scorer_name}: "
                f"{sr.result.reason or 'No explanation provided'}"
            )
            if sr.result.details:
                lines.append(f"    details: {sr.result.details!r}")
        raise AssertionError("\n".join(lines))
    return result
