from __future__ import annotations

import time
from dataclasses import dataclass, field

import click

from .scorers.base import ScoreResult


@dataclass
class ScorerResult:
    scorer_name: str
    result: ScoreResult


@dataclass
class CaseResult:
    case_name: str
    scorer_results: list[ScorerResult] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(sr.result.passed for sr in self.scorer_results)

    @property
    def score(self) -> float:
        if not self.scorer_results:
            return 1.0
        return sum(sr.result.score for sr in self.scorer_results) / len(self.scorer_results)

    @property
    def failures(self) -> list[ScorerResult]:
        return [sr for sr in self.scorer_results if not sr.result.passed]


@dataclass
class Report:
    case_results: list[CaseResult] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def passed(self) -> bool:
        return all(cr.passed for cr in self.case_results)

    @property
    def num_passed(self) -> int:
        return sum(1 for cr in self.case_results if cr.passed)

    @property
    def num_failed(self) -> int:
        return len(self.case_results) - self.num_passed

    @property
    def overall_score(self) -> float:
        if not self.case_results:
            return 1.0
        return sum(cr.score for cr in self.case_results) / len(self.case_results)

    def print(self, verbose: bool = False) -> None:
        _print_report(self, verbose=verbose)


# ── Rendering ────────────────────────────────────────────────────────────────

_PASS = click.style("PASS", fg="green", bold=True)
_FAIL = click.style("FAIL", fg="red", bold=True)


def _print_report(report: Report, verbose: bool = False) -> None:
    click.echo()
    click.echo(click.style("=" * 60, fg="bright_black"))
    click.echo(click.style("  agentdog results", bold=True))
    click.echo(click.style("=" * 60, fg="bright_black"))

    for cr in report.case_results:
        status = _PASS if cr.passed else _FAIL
        tag_str = f"  [{', '.join(cr.tags)}]" if cr.tags else ""
        click.echo(f"\n  {status}  {cr.case_name}{tag_str}  (score: {cr.score:.2f})")

        if verbose or not cr.passed:
            for sr in cr.scorer_results:
                icon = click.style("ok", fg="green") if sr.result.passed else click.style("!!", fg="red")
                line = f"       [{icon}] {sr.scorer_name}"
                if sr.result.reason:
                    line += f"  - {sr.result.reason}"
                click.echo(line)

    click.echo()
    click.echo(click.style("-" * 60, fg="bright_black"))

    summary_color = "green" if report.passed else "red"
    click.echo(
        click.style(
            f"  {report.num_passed}/{len(report.case_results)} cases passed"
            f"  |  overall score: {report.overall_score:.2f}"
            f"  |  {report.elapsed_ms:.0f}ms",
            fg=summary_color,
            bold=True,
        )
    )
    click.echo(click.style("=" * 60, fg="bright_black"))
    click.echo()
