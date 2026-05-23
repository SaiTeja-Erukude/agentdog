# agentdog
Lightweight evaluation toolkit for AI agents. `pytest` for agent behavior  —  test tool use, grounding, safety, and efficiency before production

```
pip install agentdog
pip install "agentdog[llm-judge]"  # for LLMJudge scorer
```

---

## Quickstart

```python
from agentdog import AgentTrace, ToolCall, TestCase, EvalRun, run
from agentdog import ContainsAnswer, UsedTools, AvoidedTools, UnderTokenLimit

trace = AgentTrace(
    input="Summarize the Q3 report.",
    output="Q3 revenue was $4.2M, up 12% YoY.",
    tool_calls=[ToolCall(name="file_search", arguments={"query": "Q3 report"})],
    retrieved_context=["Q3 revenue was $4.2M, growth 12% year over year."],
    total_tokens=620,
)

case = TestCase(
    name="q3-summary",
    tags=["rag"],
    scorers=[
        ContainsAnswer(["4.2M", "12%"]),
        UsedTools(["file_search"]),
        AvoidedTools(["send_email"]),
        UnderTokenLimit(max_tokens=1000),
    ],
)

report = run([EvalRun(case=case, trace=trace)])
report.print(verbose=True)
```

---

## CLI

Define an `evals()` function in any Python file that returns `list[EvalRun]`, then:

```bash
agentdog run my_evals.py             # run all cases
agentdog run my_evals.py -v          # verbose: show scorer details for passing cases
agentdog run my_evals.py --tag rag   # filter by tag
agentdog run my_evals.py --json-out report.json  # machine-readable output
agentdog inspect trace.json          # pretty-print a trace file
```

Exit code is `0` on full pass, `1` on any failure — CI-friendly by default.

---

## Scorers

| Category | Scorers |
|---|---|
| **Answer** | `ContainsAnswer` `ExactAnswer` `RegexAnswer` `ForbiddenContent` `AnswerNotEmpty` |
| **Tools** | `UsedTools` `AvoidedTools` `ToolCallOrder` `MaxToolCalls` `ToolArgContains` `ToolArgEquals` |
| **Grounding** | `GroundedInContext` `CitedSource` `NoContextHallucination` |
| **Safety** | `NoSensitiveDataLeaked` `NoRiskyActionTaken` `PromptInjectionResisted` |
| **Efficiency** | `UnderTokenLimit` `UnderCostLimit` `UnderLatencyLimit` `MaxRetries` |
| **LLM Judge** | `LLMJudge` — use only when deterministic checks aren't enough |

---

## Trace schema

```python
AgentTrace(
    input: str,
    output: str,
    tool_calls: list[ToolCall],        # name, arguments, output, error, latency_ms
    retrieved_context: list[str],
    total_tokens: int | None,
    total_cost_usd: float | None,
    total_latency_ms: float | None,
    num_retries: int,
    metadata: dict,
)
```

Load/save:

```python
trace = AgentTrace.from_json("trace.json")
trace.to_json("trace.json")
```

---

## Custom scorer

```python
from agentdog.scorers.base import Scorer, ScoreResult

class AnswerStartsWith(Scorer):
    def __init__(self, prefix: str):
        self.prefix = prefix

    def score(self, trace) -> ScoreResult:
        passed = trace.output.startswith(self.prefix)
        return ScoreResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reason=f"Expected output to start with {self.prefix!r}",
        )
```

---

## Example

See [`examples/sample_evals.py`](examples/sample_evals.py) for a complete working example covering RAG, safety, and prompt injection.

---

## Author

**Sai Teja Erukude**  
[GitHub](https://github.com/SaiTeja-Erukude) · [agentdog](https://github.com/SaiTeja-Erukude/agentdog)

Licensed under the [MIT License](LICENSE).
