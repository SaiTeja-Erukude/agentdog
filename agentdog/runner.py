from __future__ import annotations

import time
from typing import Sequence

from .case import EvalRun
from .report import CaseResult, Report, ScorerResult


def run(eval_runs: Sequence[EvalRun], verbose: bool = False) -> Report:
    """
    Execute all scorers for each EvalRun and return a Report.

    Args:
        eval_runs: Sequence of (TestCase, AgentTrace) pairs to evaluate.
        verbose:   If True, print detailed scorer output for passing cases too.

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
                from .scorers.base import ScoreResult

                result = ScoreResult(
                    passed=False,
                    score=0.0,
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

    elapsed_ms = (time.perf_counter() - start) * 1000
    return Report(case_results=case_results, elapsed_ms=elapsed_ms)
