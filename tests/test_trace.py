import json
import tempfile
from pathlib import Path

from agentdog.trace import AgentTrace, ToolCall


def test_tool_names():
    trace = AgentTrace(
        input="q",
        output="a",
        tool_calls=[
            ToolCall(name="search", arguments={}),
            ToolCall(name="summarize", arguments={}),
        ],
    )
    assert trace.tool_names == ["search", "summarize"]


def test_get_tool_calls():
    trace = AgentTrace(
        input="q",
        output="a",
        tool_calls=[
            ToolCall(name="search", arguments={"q": "foo"}),
            ToolCall(name="search", arguments={"q": "bar"}),
            ToolCall(name="other", arguments={}),
        ],
    )
    calls = trace.get_tool_calls("search")
    assert len(calls) == 2
    assert all(c.name == "search" for c in calls)


def test_tool_outputs():
    trace = AgentTrace(
        input="q",
        output="a",
        tool_calls=[
            ToolCall(name="search", arguments={}, output="result text"),
            ToolCall(name="other", arguments={}, output=None),
        ],
    )
    assert trace.tool_outputs == ["result text"]


def test_round_trip_json():
    trace = AgentTrace(
        input="hello",
        output="world",
        tool_calls=[ToolCall(name="t", arguments={"k": "v"}, output="out")],
        retrieved_context=["ctx"],
        total_tokens=100,
        total_cost_usd=0.01,
        total_latency_ms=500.0,
        num_retries=1,
        metadata={"model": "gpt-4o"},
    )
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
    trace.to_json(path)
    loaded = AgentTrace.from_json(path)
    assert loaded.input == trace.input
    assert loaded.output == trace.output
    assert loaded.tool_calls[0].name == "t"
    assert loaded.total_tokens == 100
    assert loaded.metadata == {"model": "gpt-4o"}


def test_from_dict_minimal():
    trace = AgentTrace.from_dict({"input": "hi", "output": "there"})
    assert trace.input == "hi"
    assert trace.tool_calls == []
    assert trace.num_retries == 0
