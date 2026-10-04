import pytest

from agentdog.scorers.efficiency import (
    MaxRetries,
    UnderCostLimit,
    UnderLatencyLimit,
    UnderTokenLimit,
)
from agentdog.trace import AgentTrace


def make_trace(**kwargs) -> AgentTrace:
    return AgentTrace(input="q", output="a", **kwargs)


class TestUnderTokenLimit:
    def test_under(self):
        assert UnderTokenLimit(1000).score(make_trace(total_tokens=500)).passed

    def test_at_limit(self):
        assert UnderTokenLimit(500).score(make_trace(total_tokens=500)).passed

    def test_over(self):
        r = UnderTokenLimit(500).score(make_trace(total_tokens=600))
        assert not r.passed

    def test_missing_tokens_skipped(self, simple_trace):
        simple_trace.total_tokens = None
        r = UnderTokenLimit(100).score(simple_trace)
        assert not r.passed
        assert r.status == "skip"
        assert r.score is None


class TestUnderCostLimit:
    def test_under(self):
        assert UnderCostLimit(0.01).score(make_trace(total_cost_usd=0.005)).passed

    def test_over(self):
        r = UnderCostLimit(0.001).score(make_trace(total_cost_usd=0.01))
        assert not r.passed

    def test_missing_cost_skipped(self, simple_trace):
        simple_trace.total_cost_usd = None
        r = UnderCostLimit(0.01).score(simple_trace)
        assert not r.passed
        assert r.status == "skip"
        assert r.score is None


class TestUnderLatencyLimit:
    def test_under(self):
        assert UnderLatencyLimit(2000).score(make_trace(total_latency_ms=1000)).passed

    def test_over(self):
        r = UnderLatencyLimit(500).score(make_trace(total_latency_ms=1500))
        assert not r.passed

    def test_missing_latency_skipped(self, simple_trace):
        simple_trace.total_latency_ms = None
        r = UnderLatencyLimit(1000).score(simple_trace)
        assert not r.passed
        assert r.status == "skip"
        assert r.score is None


class TestMaxRetries:
    def test_no_retries(self, simple_trace):
        assert MaxRetries(max_retries=0).score(simple_trace).passed

    def test_at_limit(self):
        assert MaxRetries(max_retries=2).score(make_trace(num_retries=2)).passed

    def test_over_limit(self):
        r = MaxRetries(max_retries=1).score(make_trace(num_retries=3))
        assert not r.passed


@pytest.mark.parametrize("scorer_type,limit", [
    (UnderTokenLimit, 100), (UnderCostLimit, 0.01), (UnderLatencyLimit, 1000),
])
@pytest.mark.parametrize("missing,passed,score", [
    ("skip", False, None), ("fail", False, 0.0),
    ("error", False, None), ("pass", True, 1.0),
])
def test_missing_measurement_policy(scorer_type, limit, missing, passed, score):
    result = scorer_type(limit, missing=missing).score(make_trace())
    assert result.status == missing
    assert result.passed is passed
    assert bool(result) is passed
    assert result.score == score
    assert result.details == {"missing_data": True, "policy": missing}


@pytest.mark.parametrize("scorer_type,limit", [
    (UnderTokenLimit, 100), (UnderCostLimit, 0.01), (UnderLatencyLimit, 1000),
])
def test_unknown_missing_policy_rejected(scorer_type, limit):
    with pytest.raises(ValueError, match="missing-data policy"):
        scorer_type(limit, missing="ignore")
