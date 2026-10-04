from __future__ import annotations

from dataclasses import dataclass, field

import click

from .scorers.base import ScoreResult, ScoreStatus


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
    def status(self) -> ScoreStatus:
        statuses = [sr.result.status for sr in self.scorer_results]
        if "error" in statuses:
            return "error"
        if "fail" in statuses:
            return "fail"
        if statuses and all(status == "skip" for status in statuses):
            return "skip"
        return "pass"

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    @property
    def score(self) -> float | None:
        if not self.scorer_results:
            return 1.0
        scores = [sr.result.score for sr in self.scorer_results if sr.result.score is not None]
        return sum(scores) / len(scores) if scores else None

    @property
    def num_evaluated(self) -> int:
        return sum(sr.result.status in ("pass", "fail") for sr in self.scorer_results)

    @property
    def num_skipped(self) -> int:
        return sum(sr.result.status == "skip" for sr in self.scorer_results)

    @property
    def num_errors(self) -> int:
        return sum(sr.result.status == "error" for sr in self.scorer_results)

    @property
    def failures(self) -> list[ScorerResult]:
        return [sr for sr in self.scorer_results if sr.result.status in ("fail", "error")]


@dataclass
class Report:
    case_results: list[CaseResult] = field(default_factory=list)
    elapsed_ms: float = 0.0
    total_cases: int | None = None

    @property
    def passed(self) -> bool:
        """Whether the run has no failures/errors; skips are non-blocking."""
        return all(cr.status in ("pass", "skip") for cr in self.case_results)

    @property
    def num_passed(self) -> int:
        return sum(1 for cr in self.case_results if cr.passed)

    @property
    def num_failed(self) -> int:
        return sum(cr.status == "fail" for cr in self.case_results)

    @property
    def num_skipped(self) -> int:
        return sum(cr.status == "skip" for cr in self.case_results)

    @property
    def num_errors(self) -> int:
        return sum(cr.status == "error" for cr in self.case_results)

    @property
    def num_scorers(self) -> int:
        return sum(len(cr.scorer_results) for cr in self.case_results)

    @property
    def num_evaluated(self) -> int:
        return sum(cr.num_evaluated for cr in self.case_results)

    @property
    def num_scorers_skipped(self) -> int:
        return sum(cr.num_skipped for cr in self.case_results)

    @property
    def num_scorer_errors(self) -> int:
        return sum(cr.num_errors for cr in self.case_results)

    @property
    def coverage(self) -> float:
        """Fraction of attempted scorer checks that produced evaluated results."""
        return self.num_evaluated / self.num_scorers if self.num_scorers else 0.0

    @property
    def num_not_run(self) -> int:
        total = self.total_cases if self.total_cases is not None else len(self.case_results)
        return total - len(self.case_results)

    @property
    def complete(self) -> bool:
        return self.num_not_run == 0

    @property
    def overall_score(self) -> float | None:
        if not self.case_results:
            return 1.0
        scores = [cr.score for cr in self.case_results if cr.score is not None]
        return sum(scores) / len(scores) if scores else None

    def print(self, verbose: bool = False) -> None:
        _print_report(self, verbose=verbose)


# ── Rendering ────────────────────────────────────────────────────────────────

_COLORS = {"pass": "green", "fail": "red", "skip": "yellow", "error": "red"}


def _score_text(score: float | None) -> str:
    return f"{score:.2f}" if score is not None else "n/a"


def _print_report(report: Report, verbose: bool = False) -> None:
    click.echo()
    click.echo(click.style("=" * 60, fg="bright_black"))
    click.echo(click.style("  agentdog results", bold=True))
    click.echo(click.style("=" * 60, fg="bright_black"))

    for cr in report.case_results:
        status = click.style(cr.status.upper(), fg=_COLORS[cr.status], bold=True)
        tag_str = f"  [{', '.join(cr.tags)}]" if cr.tags else ""
        click.echo(f"\n  {status}  {cr.case_name}{tag_str}  (score: {_score_text(cr.score)})")

        for sr in cr.scorer_results:
            if verbose or not cr.passed or sr.result.status != "pass":
                icon = click.style(sr.result.status.upper(), fg=_COLORS[sr.result.status])
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
            f"  |  {report.num_failed} failed, {report.num_skipped} skipped, {report.num_errors} errors"
            f"  |  overall score: {_score_text(report.overall_score)}"
            f"  |  evaluated checks: {report.num_evaluated}/{report.num_scorers}"
            f" ({report.coverage:.0%})"
            f"  |  {report.elapsed_ms:.0f}ms",
            fg=summary_color,
            bold=True,
        )
    )
    click.echo(
        f"  Scorer checks: {report.num_scorers_skipped} skipped, "
        f"{report.num_scorer_errors} errors | cases not run: {report.num_not_run}"
    )
    click.echo(click.style("=" * 60, fg="bright_black"))
    click.echo()
