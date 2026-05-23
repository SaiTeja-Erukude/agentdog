import pytest

from agentdog.scorers.tools import (
    AvoidedTools,
    MaxToolCalls,
    ToolArgContains,
    ToolArgEquals,
    ToolCallOrder,
    UsedTools,
)
from agentdog.trace import AgentTrace, ToolCall


def make_trace(*tool_names: str, **kwargs) -> AgentTrace:
    return AgentTrace(
        input="q",
        output="a",
        tool_calls=[ToolCall(name=n, arguments=kwargs.get(n, {})) for n in tool_names],
    )


class TestUsedTools:
    def test_all_used(self, rag_trace):
        r = UsedTools(["file_search"]).score(rag_trace)
        assert r.passed

    def test_missing_tool(self, rag_trace):
        r = UsedTools(["file_search", "web_search"]).score(rag_trace)
        assert not r.passed
        assert r.score == 0.5

    def test_empty_trace(self, simple_trace):
        r = UsedTools(["file_search"]).score(simple_trace)
        assert not r.passed


class TestAvoidedTools:
    def test_no_violation(self, rag_trace):
        r = AvoidedTools(["send_email", "delete_file"]).score(rag_trace)
        assert r.passed

    def test_violation(self, multi_tool_trace):
        r = AvoidedTools(["send_email"]).score(multi_tool_trace)
        assert not r.passed

    def test_partial_violation(self, multi_tool_trace):
        r = AvoidedTools(["send_email", "noop"]).score(multi_tool_trace)
        assert not r.passed
        assert r.score == 0.5


class TestToolCallOrder:
    def test_correct_order(self, multi_tool_trace):
        r = ToolCallOrder(["search_flights", "book_flight", "send_email"]).score(multi_tool_trace)
        assert r.passed

    def test_wrong_order(self, multi_tool_trace):
        r = ToolCallOrder(["book_flight", "search_flights"]).score(multi_tool_trace)
        assert not r.passed

    def test_subsequence(self, multi_tool_trace):
        r = ToolCallOrder(["search_flights", "send_email"]).score(multi_tool_trace)
        assert r.passed


class TestMaxToolCalls:
    def test_under_limit(self, rag_trace):
        assert MaxToolCalls(max_calls=3).score(rag_trace).passed

    def test_at_limit(self, multi_tool_trace):
        assert MaxToolCalls(max_calls=3).score(multi_tool_trace).passed

    def test_over_limit(self, multi_tool_trace):
        r = MaxToolCalls(max_calls=2).score(multi_tool_trace)
        assert not r.passed

    def test_no_calls(self, simple_trace):
        assert MaxToolCalls(max_calls=0).score(simple_trace).passed


class TestToolArgContains:
    def test_match(self, rag_trace):
        r = ToolArgContains("file_search", "query", "Q3").score(rag_trace)
        assert r.passed

    def test_no_match(self, rag_trace):
        r = ToolArgContains("file_search", "query", "annual").score(rag_trace)
        assert not r.passed

    def test_tool_not_called(self, simple_trace):
        r = ToolArgContains("file_search", "query", "Q3").score(simple_trace)
        assert not r.passed


class TestToolArgEquals:
    def test_match(self):
        trace = AgentTrace(
            input="q",
            output="a",
            tool_calls=[ToolCall(name="send_email", arguments={"to": "bob@example.com"})],
        )
        r = ToolArgEquals("send_email", "to", "bob@example.com").score(trace)
        assert r.passed

    def test_no_match(self):
        trace = AgentTrace(
            input="q",
            output="a",
            tool_calls=[ToolCall(name="send_email", arguments={"to": "alice@example.com"})],
        )
        r = ToolArgEquals("send_email", "to", "bob@example.com").score(trace)
        assert not r.passed
