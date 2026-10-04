"""Run with: python -m pytest examples/test_agent_behavior.py -q"""
import pytest

from agentdog import AgentTrace, ContainsAnswer, UnderTokenLimit, assert_evaluates


@pytest.mark.parametrize("output", ["Paris", "The capital of France is Paris."])
def test_capital_answer(output):
    trace = AgentTrace(input="What is the capital of France?", output=output, total_tokens=40)
    assert_evaluates(
        trace,
        [ContainsAnswer(["Paris"]), UnderTokenLimit(100)],
        name="france-capital",
    )
