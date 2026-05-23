from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A single tool invocation within an agent run."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    error: str | None = None
    latency_ms: float | None = None
    tokens: int | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "output": self.output,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "tokens": self.tokens,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ToolCall":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AgentTrace:
    """
    Canonical trace of a single agent run.

    Framework adapters should convert their native trace format into this
    schema before passing it to scorers or the runner.
    """

    input: str
    output: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    retrieved_context: list[str] = field(default_factory=list)
    total_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_cost_usd: float | None = None
    total_latency_ms: float | None = None
    num_retries: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_names(self) -> list[str]:
        return [tc.name for tc in self.tool_calls]

    @property
    def tool_outputs(self) -> list[str]:
        return [str(tc.output) for tc in self.tool_calls if tc.output is not None]

    def get_tool_calls(self, name: str) -> list[ToolCall]:
        return [tc for tc in self.tool_calls if tc.name == name]

    def to_dict(self) -> dict:
        return {
            "input": self.input,
            "output": self.output,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "retrieved_context": self.retrieved_context,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_cost_usd": self.total_cost_usd,
            "total_latency_ms": self.total_latency_ms,
            "num_retries": self.num_retries,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentTrace":
        tool_calls = [ToolCall.from_dict(tc) for tc in d.get("tool_calls", [])]
        return cls(
            input=d["input"],
            output=d["output"],
            tool_calls=tool_calls,
            retrieved_context=d.get("retrieved_context", []),
            total_tokens=d.get("total_tokens"),
            prompt_tokens=d.get("prompt_tokens"),
            completion_tokens=d.get("completion_tokens"),
            total_cost_usd=d.get("total_cost_usd"),
            total_latency_ms=d.get("total_latency_ms"),
            num_retries=d.get("num_retries", 0),
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, path: str) -> "AgentTrace":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
