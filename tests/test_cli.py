import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from agentdog import AgentTrace, EvalRun, ScoreResult, Scorer, TestCase
from agentdog.cli import main


class CountingScorer(Scorer):
    def __init__(self, status):
        self.status = status
        self.calls = 0

    def score(self, trace):
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("Scorer evaluated twice")
        return ScoreResult(
            self.status == "pass", 1.0 if self.status == "pass" else 0.0,
            reason=f"result: {self.status}", details={"calls": self.calls}, status=self.status,
        )


@pytest.mark.parametrize("statuses,expected_calls,exit_code,complete", [
    (["pass", "fail", "pass"], [1, 1, 0], 1, False),
    (["pass", "error", "pass"], [1, 1, 0], 1, False),
    (["pass", "pass", "pass"], [1, 1, 1], 0, True),
    (["skip", "pass", "pass"], [1, 1, 1], 0, True),
    (["pass", "pass", "fail"], [1, 1, 1], 1, True),
])
def test_fail_fast_scores_once_and_exports(
    tmp_path, monkeypatch, statuses, expected_calls, exit_code, complete
):
    scorers = [CountingScorer(status) for status in statuses]
    runs = [
        EvalRun(TestCase(f"case-{i}", [scorer]), AgentTrace("input", "output"))
        for i, scorer in enumerate(scorers)
    ]
    module = tmp_path / "evals.py"
    module.write_text("# module loading stubbed below", encoding="utf-8")
    monkeypatch.setattr("agentdog.cli._load_module", lambda path: SimpleNamespace(evals=lambda: runs))
    output_path = tmp_path / "report.json"
    result = CliRunner().invoke(main, [
        "run", str(module), "--fail-fast", "--json-out", str(output_path),
    ])
    assert result.exit_code == exit_code, result.output
    assert [scorer.calls for scorer in scorers] == expected_calls
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["complete"] is complete
    assert data["total_cases"] == 3
    assert data["num_not_run"] == expected_calls.count(0)
    assert len(data["cases"]) == sum(expected_calls)
    for case, status in zip(data["cases"], statuses):
        assert case["status"] == status
        assert case["scorers"][0]["status"] == status
        assert case["scorers"][0]["details"] == {"calls": 1}
        if status in ("skip", "error"):
            assert case["score"] is None
            assert case["scorers"][0]["score"] is None


def test_cli_json_exposes_missing_metric_coverage(tmp_path):
    module = tmp_path / "evals.py"
    module.write_text(
        "from agentdog import AgentTrace, EvalRun, TestCase, ContainsAnswer, UnderTokenLimit\n"
        "def evals():\n"
        "    return [EvalRun(TestCase('mixed', [ContainsAnswer(['answer']), "
        "UnderTokenLimit(10)]), AgentTrace('q', 'answer'))]\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "report.json"
    result = CliRunner().invoke(main, ["run", str(module), "--json-out", str(output_path)])
    assert result.exit_code == 0, result.output
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["num_evaluated"] == 1
    assert data["num_scorers"] == 2
    assert data["num_scorers_skipped"] == 1
    assert data["coverage"] == 0.5
    assert data["overall_score"] == 1.0
