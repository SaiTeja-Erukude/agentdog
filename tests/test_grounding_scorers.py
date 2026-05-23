import pytest

from agentdog.scorers.grounding import CitedSource, GroundedInContext, NoContextHallucination
from agentdog.trace import AgentTrace


def make_trace(output: str, context: list[str] | None = None) -> AgentTrace:
    return AgentTrace(input="q", output=output, retrieved_context=context or [])


class TestGroundedInContext:
    def test_well_grounded(self, rag_trace):
        r = GroundedInContext(threshold=0.2).score(rag_trace)
        assert r.passed
        assert r.score >= 0.2

    def test_no_context(self, simple_trace):
        r = GroundedInContext().score(simple_trace)
        assert not r.passed

    def test_unrelated_answer(self):
        trace = make_trace(
            "The moon is made of cheese.",
            context=["Q3 revenue was 4.2M."],
        )
        r = GroundedInContext(threshold=0.8).score(trace)
        assert not r.passed

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            GroundedInContext(threshold=0.0)


class TestCitedSource:
    def test_bracket_citation(self):
        r = CitedSource().score(make_trace("Revenue was $4.2M [1]."))
        assert r.passed

    def test_according_to(self):
        r = CitedSource().score(make_trace("According to the report, revenue grew."))
        assert r.passed

    def test_no_citation(self):
        r = CitedSource().score(make_trace("Revenue grew last quarter."))
        assert not r.passed

    def test_custom_pattern(self):
        r = CitedSource(custom_patterns=[r"see doc"]).score(make_trace("See doc for details."))
        assert r.passed


class TestNoContextHallucination:
    def test_grounded_answer(self, rag_trace):
        r = NoContextHallucination().score(rag_trace)
        assert r.passed

    def test_no_context(self, simple_trace):
        r = NoContextHallucination().score(simple_trace)
        assert not r.passed

    def test_empty_output(self):
        trace = make_trace("", context=["some context here"])
        r = NoContextHallucination().score(trace)
        assert r.passed
