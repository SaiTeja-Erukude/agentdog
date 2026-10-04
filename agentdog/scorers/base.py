from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from agentdog.trace import AgentTrace


ScoreStatus = Literal["pass", "fail", "skip", "error"]


@dataclass
class ScoreResult:
    passed: bool
    score: float | None  # None means the check was not evaluated
    reason: str = ""
    details: dict = field(default_factory=dict)
    status: ScoreStatus | None = None

    def __post_init__(self) -> None:
        # Existing custom scorers can continue to supply passed and score only.
        if self.status is None:
            self.status = "pass" if self.passed else "fail"
        if self.status not in ("pass", "fail", "skip", "error"):
            raise ValueError(f"Unknown score status: {self.status!r}")
        self.passed = self.status == "pass"
        if self.status in ("skip", "error"):
            self.score = None
        elif self.score is None:
            raise ValueError("Evaluated results must have a numeric score")

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
