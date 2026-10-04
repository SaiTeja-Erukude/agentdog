import pytest

from agentdog import AgentTrace, EvalRun, ScoreResult, Scorer, TestCase, run
from agentdog.scorers import ContainsAnswer, UnderTokenLimit


class ErrorScorer(Scorer):
    def score(self, trace):
        raise RuntimeError("judge unavailable")


def make_run(name, *scorers):
    return EvalRun(TestCase(name, list(scorers)), AgentTrace(input="q", output="answer"))


@pytest.mark.parametrize("passed,status", [(True, "pass"), (False, "fail")])
def test_custom_scorer_compatibility(passed, status):
    result = ScoreResult(passed, 0.7, "reason", {"evidence": "text"})
    assert result.status == status
    assert result.passed is passed
    assert result.score == 0.7


@pytest.mark.parametrize("status", ["skip", "error"])
def test_unevaluated_results_cannot_be_passes(status):
    result = ScoreResult(True, 1.0, status=status)
    assert not result.passed
    assert not result
    assert result.score is None


def test_unknown_result_status_rejected():
    with pytest.raises(ValueError, match="Unknown score status"):
        ScoreResult(True, 1.0, status="pending")


def test_coverage_and_averages_exclude_skips_and_errors():
    report = run([
        make_run("mixed", ContainsAnswer(["answer", "missing"]), UnderTokenLimit(10)),
        make_run("skipped", UnderTokenLimit(10)),
        make_run("error", ErrorScorer()),
    ])
    assert [cr.status for cr in report.case_results] == ["fail", "skip", "error"]
    assert report.case_results[0].score == 0.5
    assert report.case_results[0].num_skipped == 1
    assert report.case_results[1].score is None
    assert report.overall_score == 0.5
    assert report.num_evaluated == 1
    assert report.num_scorers == 4
    assert report.num_scorers_skipped == 2
    assert report.num_scorer_errors == 1
    assert report.coverage == 0.25
    assert report.num_failed == report.num_skipped == report.num_errors == 1


def test_skips_are_visible_and_nonblocking():
    report = run([make_run("skipped", UnderTokenLimit(10))])
    assert report.passed
    assert not report.case_results[0].passed
    assert report.num_passed == 0
    assert report.num_skipped == 1
    assert report.overall_score is None
    assert report.coverage == 0.0
    assert report.case_results[0].failures == []


def test_terminal_shows_skip_in_mixed_passing_case(capsys):
    report = run([make_run("mixed", ContainsAnswer(["answer"]), UnderTokenLimit(10))])
    report.print()
    output = capsys.readouterr().out
    assert "PASS  mixed" in output
    assert "[SKIP] UnderTokenLimit" in output
    assert "evaluated checks: 1/2 (50%)" in output


def test_terminal_handles_only_errors_and_skips(capsys):
    report = run([make_run("skipped", UnderTokenLimit(10)), make_run("broken", ErrorScorer())])
    report.print()
    output = capsys.readouterr().out
    assert "SKIP  skipped" in output
    assert "ERROR  broken" in output
    assert "overall score: n/a" in output
    assert "1 errors" in output
