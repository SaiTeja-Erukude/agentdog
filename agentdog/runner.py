from __future__ import annotations

import time
from typing import Sequence

from .case import EvalRun
from .report import CaseResult, Report, ScorerResult
from .scorers.base import ScoreResult


def run(
    eval_runs: Sequence[EvalRun], verbose: bool = False, *, fail_fast: bool = False
) -> Report:
    """
    Execute all scorers for each EvalRun and return a Report.

    Args:
        eval_runs: Sequence of (TestCase, AgentTrace) pairs to evaluate.
        verbose:   Retained for compatibility; use Report.print(verbose=True)
                   to control detailed output.
        fail_fast: Stop after the first case with a failure or scorer error.

    Returns:
        Report with per-case and aggregate results.
    """
    start = time.perf_counter()
    case_results: list[CaseResult] = []

    for er in eval_runs:
        scorer_results: list[ScorerResult] = []
        for scorer in er.case.scorers:
            try:
                result = scorer.score(er.trace)
            except Exception as exc:
                result = ScoreResult(
                    passed=False,
                    score=None,
                    status="error",
                    reason=f"Scorer raised exception: {exc}",
                )
            scorer_results.append(ScorerResult(scorer_name=scorer.name, result=result))

        case_results.append(
            CaseResult(
                case_name=er.case.name,
                scorer_results=scorer_results,
                tags=er.case.tags,
            )
        )
        if fail_fast and case_results[-1].status in ("fail", "error"):
            break

    elapsed_ms = (time.perf_counter() - start) * 1000
    return Report(
        case_results=case_results, elapsed_ms=elapsed_ms, total_cases=len(eval_runs)
    )
