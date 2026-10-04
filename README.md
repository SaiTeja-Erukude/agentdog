# agentdog

Lightweight evaluation toolkit for AI agents. Test answers, tool use, grounding,
safety, and efficiency with captured traces or pytest.

## Install

```bash
pip install agentdog
pip install "agentdog[llm-judge]"  # optional LLM judging
```

## Quickstart

```python
from agentdog import AgentTrace, EvalRun, TestCase, ToolCall, run
from agentdog import ContainsAnswer, UsedTools, UnderTokenLimit

trace = AgentTrace(
    input="Summarize the Q3 report.",
    output="Revenue was $4.2M, up 12%.",
    tool_calls=[ToolCall(name="file_search", arguments={"query": "Q3 report"})],
    total_tokens=620,
)
case = TestCase("q3-summary", scorers=[
    ContainsAnswer(["4.2M", "12%"]),
    UsedTools(["file_search"]),
    UnderTokenLimit(1000),
])

report = run([EvalRun(case=case, trace=trace)])
report.print()
```

## CLI

Define `evals()` returning a list of `EvalRun` objects, then:

```bash
agentdog run my_evals.py -v
agentdog run my_evals.py --tag rag --fail-fast --json-out report.json
agentdog inspect trace.json
```

Exit codes: `0` for no failures/errors, `1` otherwise. Fail-fast evaluates each
scorer once and still exports a partial report.

## pytest

```python
from agentdog import AgentTrace, ContainsAnswer, assert_evaluates

def test_capital():
    trace = AgentTrace(input="Capital of France?", output="Paris")
    assert_evaluates(trace, [ContainsAnswer(["Paris"])])
```

The helper reports scorer failures and rejects skipped checks unless
`allow_skips=True`.

## Scorers

| Category | Scorers |
| --- | --- |
| Answer | `ContainsAnswer`, `ExactAnswer`, `RegexAnswer`, `ForbiddenContent`, `AnswerNotEmpty` |
| Tools | `UsedTools`, `AvoidedTools`, `ToolCallOrder`, `MaxToolCalls`, `ToolArgContains`, `ToolArgEquals` |
| Grounding | `GroundedInContext`, `CitedSource`, `NoContextHallucination` |
| Safety | `NoSensitiveDataLeaked`, `NoRiskyActionTaken`, `PromptInjectionResisted` |
| Efficiency | `UnderTokenLimit`, `UnderCostLimit`, `UnderLatencyLimit`, `MaxRetries` |
| LLM judge | `LLMJudge` |

Results have `pass`, `fail`, `skip`, or `error` status. Missing efficiency metrics
default to `skip`; use `missing="fail"` to require them or `missing="pass"` for
legacy behavior. Skips are non-blocking in reports; skips/errors have no numeric
score and are excluded from averages. Reports include evaluated coverage.

Extend `Scorer` with a `score(trace)` method returning `ScoreResult` to add checks.

See [sample evaluations](examples/sample_evals.py), the
[pytest example](examples/test_agent_behavior.py), and
[CI/release setup](.github/README.md).

Created by **Sai Teja Erukude**. [MIT License](LICENSE).
