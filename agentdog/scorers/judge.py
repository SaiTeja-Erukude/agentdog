from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ScoreResult, Scorer

if TYPE_CHECKING:
    from agentdog.trace import AgentTrace

_OPENAI_MISSING_MSG = (
    "LLMJudge requires openai. Install with: pip install agentdog[llm-judge]"
)

_DEFAULT_PROMPT = """\
You are an objective evaluator. Given the following agent input and output, \
assess whether the output satisfies the criterion below.

Criterion: {criteria}

Agent input:
{input}

Agent output:
{output}

Respond with a JSON object:
{{
  "passed": true or false,
  "score": a float between 0.0 and 1.0,
  "reason": "one-sentence explanation"
}}
"""


class LLMJudge(Scorer):
    """
    Use an LLM to evaluate a subjective criterion against the agent's output.

    Requires: pip install agentdog[llm-judge]

    Use this only when deterministic scorers are not sufficient (e.g. helpfulness,
    tone, completeness, reasoning quality).
    """

    def __init__(
        self,
        criteria: str,
        model: str = "gpt-4o-mini",
        prompt_template: str = _DEFAULT_PROMPT,
        passing_threshold: float = 0.5,
    ):
        self.criteria = criteria
        self.model = model
        self.prompt_template = prompt_template
        self.passing_threshold = passing_threshold

    def score(self, trace: "AgentTrace") -> ScoreResult:
        try:
            import json as _json

            import openai
        except ImportError:
            raise ImportError(_OPENAI_MISSING_MSG)

        client = openai.OpenAI()
        prompt = self.prompt_template.format(
            criteria=self.criteria,
            input=trace.input,
            output=trace.output,
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = response.choices[0].message.content or "{}"
        try:
            parsed = _json.loads(raw)
        except _json.JSONDecodeError:
            return ScoreResult(
                passed=False,
                score=0.0,
                reason=f"LLM judge returned invalid JSON: {raw[:200]}",
            )

        score_val = float(parsed.get("score", 0.0))
        passed = parsed.get("passed", score_val >= self.passing_threshold)
        return ScoreResult(
            passed=bool(passed),
            score=score_val,
            reason=parsed.get("reason", ""),
            details={"model": self.model, "criteria": self.criteria},
        )
