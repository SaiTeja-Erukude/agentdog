from pathlib import Path
import subprocess
import sys

import pytest

from agentdog import AgentTrace, ContainsAnswer, ScoreResult, Scorer, UnderTokenLimit
from agentdog import assert_evaluates
from agentdog.testing import assert_evaluates as _testing_helper


class ErrorScorer(Scorer):
    def score(self, trace):
        raise RuntimeError("judge unavailable")


class CountingScorer(Scorer):
    def __init__(self):
        self.calls = 0

    def score(self, trace):
        self.calls += 1
        return ScoreResult(True, 1.0)


def test_helper_returns_case_result_and_evaluates_once():
    scorer = CountingScorer()
    result = assert_evaluates(AgentTrace("q", "answer"), [scorer], name="answer case")
    assert result.passed
    assert result.case_name == "answer case"
    assert scorer.calls == 1
    assert _testing_helper is assert_evaluates


def test_failure_lists_all_failed_scorers_and_errors():
    with pytest.raises(AssertionError) as error:
        assert_evaluates(
            AgentTrace("q", "answer"),
            [ContainsAnswer(["missing"]), ErrorScorer(), UnderTokenLimit(10)],
            name="regression case",
        )
    message = str(error.value)
    assert "regression case: 3 scorer check(s) did not pass" in message
    assert "[FAIL] ContainsAnswer: Missing from output" in message
    assert "[ERROR] ErrorScorer: Scorer raised exception: judge unavailable" in message
    assert "[SKIP] UnderTokenLimit" in message
    assert "details:" in message


def test_missing_measurement_requires_explicit_skip_opt_in():
    trace = AgentTrace("q", "answer")
    with pytest.raises(AssertionError, match=r"\[SKIP\] UnderTokenLimit"):
        assert_evaluates(trace, [UnderTokenLimit(10)])
    result = assert_evaluates(trace, [UnderTokenLimit(10)], allow_skips=True)
    assert result.status == "skip"
    assert result.score is None


@pytest.mark.parametrize("scorer", [ContainsAnswer(["missing"]), ErrorScorer()])
def test_allow_skips_never_hides_failures_or_errors(scorer):
    with pytest.raises(AssertionError):
        assert_evaluates(AgentTrace("q", "answer"), [scorer], allow_skips=True)


def test_empty_scorer_list_rejected():
    with pytest.raises(ValueError, match="at least one scorer"):
        assert_evaluates(AgentTrace("q", "answer"), [])


def test_helper_in_parametrized_pytest_test(tmp_path):
    test_file = tmp_path / "test_agent_example.py"
    test_file.write_text(
        "import pytest\n"
        "from agentdog import AgentTrace, ContainsAnswer, assert_evaluates\n"
        "@pytest.mark.parametrize('output', ['Paris', 'London'])\n"
        "def test_capital(output):\n"
        "    assert_evaluates(AgentTrace('Capital of France?', output), "
        "[ContainsAnswer(['Paris'])], name='capital')\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-q", "--tb=short"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        timeout=30,
    )
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "1 failed, 1 passed" in completed.stdout
    assert "[FAIL] ContainsAnswer: Missing from output" in completed.stdout
    assert "capital: 1 scorer check(s) did not pass" in completed.stdout
