from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import click

from .case import EvalRun
from .runner import run
from .trace import AgentTrace


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("_agentdog_module", path)
    if spec is None or spec.loader is None:
        raise click.ClickException(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod


@click.group()
def main():
    """agentdog — lightweight evaluation for AI agents."""


@main.command()
@click.argument("module", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--verbose", "-v", is_flag=True, help="Show scorer details for passing cases too.")
@click.option("--fail-fast", is_flag=True, help="Stop after the first failing case.")
@click.option("--tag", "tags", multiple=True, help="Only run cases matching these tags.")
@click.option("--json-out", type=click.Path(dir_okay=False, path_type=Path), help="Write JSON report to file.")
def run_cmd(module: Path, verbose: bool, fail_fast: bool, tags: tuple[str, ...], json_out: Path | None):
    """
    Run evaluations defined in MODULE.

    MODULE must be a Python file that exposes an `evals()` function returning
    a list of EvalRun objects.

    \b
    Example module (my_evals.py):
        from agentdog import EvalRun, TestCase, AgentTrace
        from agentdog.scorers import ContainsAnswer, UsedTools

        def evals():
            trace = AgentTrace(input="What is 2+2?", output="The answer is 4.")
            case  = TestCase("basic-math", scorers=[ContainsAnswer(["4"])])
            return [EvalRun(case=case, trace=trace)]
    """
    mod = _load_module(module)
    if not hasattr(mod, "evals"):
        raise click.ClickException(f"{module} must define an evals() function")

    eval_runs: list[EvalRun] = mod.evals()

    if tags:
        eval_runs = [er for er in eval_runs if set(er.case.tags) & set(tags)]
        if not eval_runs:
            click.echo(f"No cases matched tags: {list(tags)}")
            return

    if fail_fast:
        filtered: list[EvalRun] = []
        for er in eval_runs:
            filtered.append(er)
            sub = run([er])
            if not sub.passed:
                report = run(filtered)
                report.print(verbose=verbose)
                sys.exit(1)
        eval_runs = filtered

    report = run(eval_runs)
    report.print(verbose=verbose)

    if json_out:
        _write_json_report(report, json_out)
        click.echo(f"JSON report written to {json_out}")

    sys.exit(0 if report.passed else 1)


@main.command()
@click.argument("trace_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def inspect(trace_file: Path):
    """Pretty-print a trace JSON file."""
    trace = AgentTrace.from_json(str(trace_file))
    click.echo(f"\nInput:   {trace.input}")
    click.echo(f"Output:  {trace.output}")
    click.echo(f"Tools:   {trace.tool_names or '(none)'}")
    click.echo(f"Context: {len(trace.retrieved_context)} chunk(s)")
    if trace.total_tokens is not None:
        click.echo(f"Tokens:  {trace.total_tokens}")
    if trace.total_cost_usd is not None:
        click.echo(f"Cost:    ${trace.total_cost_usd:.4f}")
    if trace.total_latency_ms is not None:
        click.echo(f"Latency: {trace.total_latency_ms:.0f}ms")
    click.echo()


def _write_json_report(report, path: Path) -> None:
    data = {
        "passed": report.passed,
        "overall_score": report.overall_score,
        "elapsed_ms": report.elapsed_ms,
        "num_passed": report.num_passed,
        "num_failed": report.num_failed,
        "cases": [
            {
                "name": cr.case_name,
                "passed": cr.passed,
                "score": cr.score,
                "tags": cr.tags,
                "scorers": [
                    {
                        "name": sr.scorer_name,
                        "passed": sr.result.passed,
                        "score": sr.result.score,
                        "reason": sr.result.reason,
                    }
                    for sr in cr.scorer_results
                ],
            }
            for cr in report.case_results
        ],
    }
    path.write_text(json.dumps(data, indent=2))
