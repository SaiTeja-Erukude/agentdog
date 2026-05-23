import pytest

from agentdog.scorers.answer import (
    AnswerNotEmpty,
    ContainsAnswer,
    ExactAnswer,
    ForbiddenContent,
    RegexAnswer,
)
from agentdog.trace import AgentTrace


def make_trace(output: str) -> AgentTrace:
    return AgentTrace(input="q", output=output)


class TestContainsAnswer:
    def test_all_present(self):
        r = ContainsAnswer(["Paris", "capital"]).score(make_trace("Paris is the capital of France."))
        assert r.passed
        assert r.score == 1.0

    def test_partial_match(self):
        r = ContainsAnswer(["Paris", "Berlin"]).score(make_trace("Paris is the capital."))
        assert not r.passed
        assert r.score == 0.5

    def test_case_insensitive_default(self):
        r = ContainsAnswer(["paris"]).score(make_trace("Paris is great."))
        assert r.passed

    def test_case_sensitive(self):
        r = ContainsAnswer(["paris"], case_sensitive=True).score(make_trace("Paris is great."))
        assert not r.passed


class TestExactAnswer:
    def test_match(self):
        r = ExactAnswer("Paris").score(make_trace("Paris"))
        assert r.passed

    def test_strip(self):
        r = ExactAnswer("Paris").score(make_trace("  Paris  "))
        assert r.passed

    def test_no_match(self):
        r = ExactAnswer("Paris").score(make_trace("London"))
        assert not r.passed
        assert r.score == 0.0


class TestRegexAnswer:
    def test_match(self):
        r = RegexAnswer(r"\d+\.\d+M").score(make_trace("Revenue was $4.2M last quarter."))
        assert r.passed

    def test_no_match(self):
        r = RegexAnswer(r"\d+\.\d+M").score(make_trace("Revenue increased significantly."))
        assert not r.passed


class TestForbiddenContent:
    def test_clean(self):
        r = ForbiddenContent(["confidential", "secret"]).score(make_trace("Public info only."))
        assert r.passed
        assert r.score == 1.0

    def test_violation(self):
        r = ForbiddenContent(["confidential"]).score(make_trace("This is confidential data."))
        assert not r.passed

    def test_partial_violation(self):
        r = ForbiddenContent(["foo", "bar"]).score(make_trace("foo is here."))
        assert not r.passed
        assert r.score == 0.5


class TestAnswerNotEmpty:
    def test_non_empty(self):
        assert AnswerNotEmpty().score(make_trace("hello")).passed

    def test_empty(self):
        assert not AnswerNotEmpty().score(make_trace("")).passed

    def test_min_chars(self):
        assert not AnswerNotEmpty(min_chars=10).score(make_trace("hi")).passed
