import pytest

from agentdog import EvalRun, TestCase, run
from agentdog.scorers.answer import ContainsAnswer, ForbiddenContent
from agentdog.scorers.base import ScoreResult, Scorer
from agentdog.trace import AgentTrace


class AlwaysPass(Scorer):
    def score(self, trace: AgentTrace) -> ScoreResult:
        return ScoreResult(passed=True, score=1.0)


class AlwaysFail(Scorer):
    def score(self, trace: AgentTrace) -> ScoreResult:
        return ScoreResult(passed=False, score=0.0, reason="intentional failure")


class BoomScorer(Scorer):
    def score(self, trace: AgentTrace) -> ScoreResult:
        raise RuntimeError("scorer exploded")


def make_run(name: str, *scorers, trace=None, tags=None):
    t = trace or AgentTrace(input="q", output="answer with keyword")
    return EvalRun(
        case=TestCase(name=name, scorers=list(scorers), tags=tags or []),
        trace=t,
    )


class TestRunner:
    def test_all_pass(self):
        report = run([make_run("a", AlwaysPass()), make_run("b", AlwaysPass())])
        assert report.passed
        assert report.num_passed == 2
        assert report.num_failed == 0
        assert report.overall_score == 1.0

    def test_one_fail(self):
        report = run([make_run("a", AlwaysPass()), make_run("b", AlwaysFail())])
        assert not report.passed
        assert report.num_passed == 1
        assert report.num_failed == 1

    def test_scorer_exception_does_not_crash(self):
        report = run([make_run("boom", BoomScorer())])
        assert not report.passed
        assert "exception" in report.case_results[0].scorer_results[0].result.reason.lower()

    def test_score_aggregation(self):
        report = run([make_run("mixed", AlwaysPass(), AlwaysFail())])
        assert report.overall_score == 0.5

    def test_empty_runs(self):
        report = run([])
        assert report.passed
        assert report.overall_score == 1.0

    def test_case_tags_preserved(self):
        report = run([make_run("tagged", AlwaysPass(), tags=["rag", "finance"])])
        assert report.case_results[0].tags == ["rag", "finance"]

    def test_elapsed_ms_positive(self, rag_trace):
        report = run([make_run("t", AlwaysPass(), trace=rag_trace)])
        assert report.elapsed_ms >= 0
