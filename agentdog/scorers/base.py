from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentdog.trace import AgentTrace


@dataclass
class ScoreResult:
    passed: bool
    score: float  # 0.0 = complete failure, 1.0 = full pass
    reason: str = ""
    details: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed


class Scorer(ABC):
    """Base class for all scorers."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def score(self, trace: "AgentTrace") -> ScoreResult:
        ...
