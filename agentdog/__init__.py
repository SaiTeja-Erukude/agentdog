"""
agentdog: lightweight evaluation toolkit for AI agents.
"""

from .case import EvalRun, TestCase
from .report import CaseResult, Report, ScorerResult
from .runner import run
from .scorers import (
    AnswerNotEmpty,
    AvoidedTools,
    CitedSource,
    ContainsAnswer,
    ExactAnswer,
    ForbiddenContent,
    GroundedInContext,
    LLMJudge,
    MaxRetries,
    MaxToolCalls,
    NoContextHallucination,
    NoRiskyActionTaken,
    NoSensitiveDataLeaked,
    PromptInjectionResisted,
    RegexAnswer,
    ScoreResult,
    Scorer,
    ToolArgContains,
    ToolArgEquals,
    ToolCallOrder,
    UnderCostLimit,
    UnderLatencyLimit,
    UnderTokenLimit,
    UsedTools,
)
from .trace import AgentTrace, ToolCall

__version__ = "0.1.0"

__all__ = [
    # trace
    "AgentTrace",
    "ToolCall",
    # case
    "TestCase",
    "EvalRun",
    # runner
    "run",
    # report
    "Report",
    "CaseResult",
    "ScorerResult",
    # scorers — base
    "Scorer",
    "ScoreResult",
    # scorers — answer
    "ContainsAnswer",
    "ExactAnswer",
    "RegexAnswer",
    "ForbiddenContent",
    "AnswerNotEmpty",
    # scorers — tools
    "UsedTools",
    "AvoidedTools",
    "ToolCallOrder",
    "MaxToolCalls",
    "ToolArgContains",
    "ToolArgEquals",
    # scorers — grounding
    "GroundedInContext",
    "CitedSource",
    "NoContextHallucination",
    # scorers — safety
    "NoSensitiveDataLeaked",
    "NoRiskyActionTaken",
    "PromptInjectionResisted",
    # scorers — efficiency
    "UnderTokenLimit",
    "UnderCostLimit",
    "UnderLatencyLimit",
    "MaxRetries",
    # scorers — llm judge
    "LLMJudge",
]

# ---------------------------------------------------------------------------
# Aspirational features — tracked here until implemented
# ---------------------------------------------------------------------------
#
# v2 candidates (medium effort, clear value):
#   - human_in_the_loop_review: structured queue for human-reviewed cases
#   - evaluation_packs: pre-built scorer bundles (rag_pack, security_pack, etc.)
#   - html_reports: rich HTML report with drill-down per case
#   - github_action: official GHA for agentdog run with PR comment integration
#   - async_runner: parallel trace evaluation for large eval sets
#
# v3 candidates (higher effort or less certain):
#   - framework_adapters: native converters for LangChain, LlamaIndex, OpenAI SDK traces
#   - model_comparison: run same cases against multiple models, diff results
#   - prompt_version_comparison: A/B eval across prompt variants
#   - trace_replay: re-run a captured trace through a new model/prompt
#   - synthetic_eval_generation: auto-generate eval cases from a prompt + schema
#   - dataset_quality_checks: flag low-quality or duplicate eval cases
#   - safety_governance_templates: pre-built packs for HIPAA, PCI, SOC2 patterns
#   - rag_advanced: retrieval-specific metrics (NDCG, MRR, recall@k)
#   - streaming_trace_support: capture and eval streaming agent runs
