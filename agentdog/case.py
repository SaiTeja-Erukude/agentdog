from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentdog.scorers.base import Scorer
    from agentdog.trace import AgentTrace


@dataclass
class TestCase:
    """Defines what to evaluate and how to evaluate it for a single agent input."""

    name: str
    scorers: list["Scorer"]
    description: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class EvalRun:
    """A test case paired with the agent trace to evaluate against."""

    case: TestCase
    trace: "AgentTrace"
