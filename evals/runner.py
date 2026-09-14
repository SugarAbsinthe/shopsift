"""Reproducible behavior evaluation for the shopping Agent.

Deterministic mode exercises the real LangGraph with controlled model, tool,
retriever, and profile-store doubles. It performs no network access and is
safe to run as a regression gate.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool

from evals.schema import DEFAULT_CASES_PATH, EvalCase, VALID_TOOLS, load_cases
from evals.quality import (
    DEFAULT_DATASET_MANIFEST_PATH,
    GateResult,
    build_run_metadata,
    default_zero_failure_gate,
    evaluation_start,
    evaluate_quality_gate,
    get_quality_gate,
    validate_dataset_artifact,
)
from src.agent.langgraph_engine import ShoppingGuideGraph


def _failure(
    failures: list[str], codes: list[str], code: str, message: str
) -> None:
    """Append a user-readable failure and one stable machine code."""
    failures.append(message)
    if code not in codes:
        codes.append(code)


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    stage: str
    tools: list[str]
    stop_reason: str
    retrieval_triggered: bool
    profile_keys: list[str]
    answer_nonempty: bool
    latency_ms: int
    failures: list[str] = field(default_factory=list)
    failure_codes: list[str] = field(default_factory=list)


@dataclass
class EvalSummary:
    total: int
    passed: int
    failed: int
    pass_rate: float
    stage_accuracy: float
    tool_boundary_accuracy: float
    stop_reason_accuracy: float
    retrieval_accuracy: float
    nonempty_answer_rate: float
    illegal_tool_calls: int
    latency_p50_ms: int
    latency_p95_ms: int
    failure_counts: dict[str, int] = field(default_factory=dict)


class ScriptedEvalLLM:
    """Small model double that follows a case's explicit tool script."""

    def __init__(self, case: EvalCase):
        self.case = case
        self.agent_calls = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, **kwargs):
        if isinstance(messages, str):
            return AIMessage(
                content=self.case.classifier_response or self.case.expected_stages[0]
            )

        call_index = self.agent_calls
        self.agent_calls += 1
        if call_index < len(self.case.scripted_tool_rounds):
            calls = [
                {
                    "name": tool_name,
                    "args": {"payload": "evaluation-fixture"},
                    "id": f"{self.case.id}-{call_index}-{tool_index}",
                    "type": "tool_call",
                }
                for tool_index, tool_name in enumerate(
                    self.case.scripted_tool_rounds[call_index]
                )
            ]
            return AIMessage(content="", tool_calls=calls)
        return AIMessage(content=self.case.deterministic_answer)


class RecordingRetriever:
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> str:
        self.calls.append((query, top_k))
        if self.should_fail:
            raise RuntimeError("deterministic retriever failure")
        return "产品101｜评测夹具商品｜价格 6999 元"


class RecordingProfileStore:
    def __init__(self):
        self.values: dict[str, str] = {}

    def serialize_profile(self, conv_id: str) -> str:
        if not self.values:
            return "(暂无画像)"
        return "\n".join(f"{key}: {value}" for key, value in sorted(self.values.items()))

    def update(self, conv_id: str, key: str, value: str, **kwargs) -> None:
        self.values[key] = value


class ToolRecorder:
    def __init__(self, failing_tools: Iterable[str] = ()):
        self.failing_tools = set(failing_tools)
        self.calls: list[str] = []

    def build_tools(self) -> list[StructuredTool]:
        return [self._build_tool(name) for name in sorted(VALID_TOOLS)]

    def _build_tool(self, name: str) -> StructuredTool:
        recorder = self

        def execute(payload: str = "evaluation-fixture") -> str:
            recorder.calls.append(name)
            if name in recorder.failing_tools:
                raise RuntimeError(f"deterministic {name} failure")
            return f"{name} completed"

        return StructuredTool.from_function(
            func=execute,
            name=name,
            description=f"Deterministic evaluation double for {name}.",
        )


def _to_history(case: EvalCase) -> list:
    history = []
    for message in case.history:
        cls = HumanMessage if message.role == "user" else AIMessage
        history.append(cls(content=message.content))
    return history


def run_deterministic_case(case: EvalCase) -> CaseResult:
    """Run one case against the production graph with local doubles."""
    llm = ScriptedEvalLLM(case)
    retriever = RecordingRetriever(case.retriever_error)
    profile_store = RecordingProfileStore()
    tool_recorder = ToolRecorder(case.failing_tools)
    started_at = time.perf_counter()
    graph = None
    result = None
    execution_error = ""
    try:
        graph = ShoppingGuideGraph(
            llm=llm,
            tools=tool_recorder.build_tools(),
            product_retriever=retriever,
            profile_store=profile_store,
            system_prompt="test {conv_id} {stage} {user_profile} {product_context}",
            stage_classifier_prompt=(
                "current={current_stage}\nmessage={user_message}\nreturn one stage"
            ),
            max_tool_rounds=case.max_tool_rounds,
        )
        result = graph.run(
            user_message=case.question,
            conv_id=f"eval-{case.id}",
            chat_history=_to_history(case),
        )
    except Exception as exc:
        # Do not expose exception details from an evaluation fixture or provider.
        execution_error = type(exc).__name__
    finally:
        if graph is not None:
            graph.close()
    latency_ms = round((time.perf_counter() - started_at) * 1000)

    if result is None:
        return CaseResult(
            case_id=case.id,
            passed=False,
            stage="",
            tools=list(tool_recorder.calls),
            stop_reason="",
            retrieval_triggered=bool(retriever.calls),
            profile_keys=sorted(profile_store.values),
            answer_nonempty=False,
            latency_ms=latency_ms,
            failures=[f"execution error: {execution_error or 'UnknownError'}"],
            failure_codes=["EXECUTION_ERROR"],
        )

    answer = ""
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage) and message.content:
            answer = str(message.content).strip()
            break

    stage = result.get("stage", "")
    stop_reason = result.get("stop_reason", "")
    retrieval_triggered = bool(retriever.calls)
    observed_tools = tool_recorder.calls
    profile_keys = sorted(profile_store.values)
    failures: list[str] = []
    failure_codes: list[str] = []

    if stage not in case.expected_stages:
        _failure(
            failures,
            failure_codes,
            "STAGE_MISMATCH",
            f"stage={stage!r}, expected one of {case.expected_stages}",
        )
    missing_tools = sorted(set(case.required_tools) - set(observed_tools))
    if missing_tools:
        _failure(
            failures,
            failure_codes,
            "REQUIRED_TOOL_MISSING",
            f"missing required tools: {missing_tools}",
        )
    forbidden_tools = sorted(set(case.forbidden_tools) & set(observed_tools))
    if forbidden_tools:
        _failure(
            failures,
            failure_codes,
            "FORBIDDEN_TOOL_CALLED",
            f"called forbidden tools: {forbidden_tools}",
        )
    if stop_reason not in case.expected_stop_reasons:
        _failure(
            failures,
            failure_codes,
            "STOP_REASON_MISMATCH",
            f"stop_reason={stop_reason!r}, expected one of {case.expected_stop_reasons}",
        )
    if retrieval_triggered != case.expected_retrieval:
        _failure(
            failures,
            failure_codes,
            "RETRIEVAL_TRIGGER_MISMATCH",
            f"retrieval_triggered={retrieval_triggered}, "
            f"expected {case.expected_retrieval}",
        )
    missing_profile_keys = sorted(set(case.expected_profile_keys) - set(profile_keys))
    if missing_profile_keys:
        _failure(
            failures,
            failure_codes,
            "PROFILE_KEY_MISSING",
            f"missing profile keys: {missing_profile_keys}",
        )
    if not answer:
        _failure(failures, failure_codes, "EMPTY_ANSWER", "final answer is empty")

    return CaseResult(
        case_id=case.id,
        passed=not failures,
        stage=stage,
        tools=list(observed_tools),
        stop_reason=stop_reason,
        retrieval_triggered=retrieval_triggered,
        profile_keys=profile_keys,
        answer_nonempty=bool(answer),
        latency_ms=latency_ms,
        failures=failures,
        failure_codes=failure_codes,
    )


def _percent(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _percentile(values: Sequence[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[rank]


def summarize(cases: Sequence[EvalCase], results: Sequence[CaseResult]) -> EvalSummary:
    if len(cases) != len(results):
        raise ValueError("case and result counts must match")
    total = len(results)
    passed = sum(result.passed for result in results)
    stage_ok = sum(
        result.stage in case.expected_stages
        for case, result in zip(cases, results)
    )
    tool_ok = sum(
        set(case.required_tools).issubset(result.tools)
        and not (set(case.forbidden_tools) & set(result.tools))
        for case, result in zip(cases, results)
    )
    stop_ok = sum(
        result.stop_reason in case.expected_stop_reasons
        for case, result in zip(cases, results)
    )
    retrieval_ok = sum(
        result.retrieval_triggered == case.expected_retrieval
        for case, result in zip(cases, results)
    )
    nonempty = sum(result.answer_nonempty for result in results)
    illegal_calls = sum(
        len(set(case.forbidden_tools) & set(result.tools))
        for case, result in zip(cases, results)
    )
    latencies = [result.latency_ms for result in results]
    failure_counts: dict[str, int] = {}
    for result in results:
        for code in result.failure_codes:
            failure_counts[code] = failure_counts.get(code, 0) + 1
    return EvalSummary(
        total=total,
        passed=passed,
        failed=total - passed,
        pass_rate=_percent(passed, total),
        stage_accuracy=_percent(stage_ok, total),
        tool_boundary_accuracy=_percent(tool_ok, total),
        stop_reason_accuracy=_percent(stop_ok, total),
        retrieval_accuracy=_percent(retrieval_ok, total),
        nonempty_answer_rate=_percent(nonempty, total),
        illegal_tool_calls=illegal_calls,
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        failure_counts=dict(sorted(failure_counts.items())),
    )


def run_deterministic(cases: Sequence[EvalCase]) -> tuple[EvalSummary, list[CaseResult]]:
    results = [run_deterministic_case(case) for case in cases]
    return summarize(cases, results), results


def write_reports(
    summary: EvalSummary,
    results: Sequence[CaseResult],
    output_dir: Path | str,
    mode: str = "deterministic",
    metadata: dict | None = None,
    gate: GateResult | None = None,
) -> tuple[Path, Path]:
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{mode}.json"
    markdown_path = report_dir / f"{mode}.md"

    json_path.write_text(
        json.dumps(
            {
                "report_schema_version": 1,
                "mode": mode,
                "metadata": metadata or {},
                "gate": gate.to_dict() if gate else None,
                "summary": asdict(summary),
                "results": [asdict(result) for result in results],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        f"# {mode.title()} Agent Evaluation",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Cases | {summary.total} |",
        f"| Passed | {summary.passed} |",
        f"| Pass rate | {summary.pass_rate:.1%} |",
        f"| Stage accuracy | {summary.stage_accuracy:.1%} |",
        f"| Tool boundary accuracy | {summary.tool_boundary_accuracy:.1%} |",
        f"| Stop reason accuracy | {summary.stop_reason_accuracy:.1%} |",
        f"| Retrieval accuracy | {summary.retrieval_accuracy:.1%} |",
        f"| Non-empty answer rate | {summary.nonempty_answer_rate:.1%} |",
        f"| Illegal tool calls | {summary.illegal_tool_calls} |",
        f"| Latency p50 / p95 | {summary.latency_p50_ms} / {summary.latency_p95_ms} ms |",
        f"| Failure codes | {', '.join(f'{key}: {value}' for key, value in summary.failure_counts.items()) or 'none'} |",
        "",
    ]
    if metadata:
        dataset = metadata.get("dataset", {})
        lines.extend([
            "## Run metadata",
            "",
            f"- Report schema: `{metadata.get('report_schema_version', 1)}`",
            f"- Dataset: `{dataset.get('dataset_id', 'ad-hoc')}` / `{dataset.get('dataset_version', 'unversioned')}`",
            f"- Dataset SHA-256: `{dataset.get('sha256', 'unknown')}`",
            f"- Code revision: `{metadata.get('code', {}).get('git_revision') or 'unknown'}`",
            f"- Worktree dirty: `{metadata.get('code', {}).get('worktree_dirty')}`",
            f"- Model/provider: `{metadata.get('runtime', {}).get('model', 'unknown')}` / `{metadata.get('runtime', {}).get('provider', 'unknown')}`",
            f"- External access: `{metadata.get('runtime', {}).get('external_access')}`",
            "",
        ])
    if gate:
        lines.extend([
            "## Quality gate",
            "",
            f"- `{gate.name}`: **{'passed' if gate.passed else 'failed'}**",
        ])
        if gate.failures:
            lines.extend(f"- {failure}" for failure in gate.failures)
        lines.append("")
    lines.extend([
        "## Failures",
        "",
    ])
    failed = [result for result in results if not result.passed]
    if failed:
        for result in failed:
            codes = f" [{', '.join(result.failure_codes)}]" if result.failure_codes else ""
            lines.append(f"- `{result.case_id}`{codes}: {'; '.join(result.failures)}")
    else:
        lines.append("No failures.")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def run_live_case(agent, case: EvalCase) -> CaseResult:
    """Run one case with configured production dependencies and clean it up."""
    conv_id = f"eval-live-{case.id}-{uuid.uuid4().hex[:8]}"
    started_at = time.perf_counter()
    result = None
    execution_failure = ""
    profile_inspection_failure = False
    cleanup_failure = False
    profile_keys: list[str] = []
    try:
        result = agent.run(
            question=case.question,
            conv_id=conv_id,
            chat_history=_to_history(case),
        )
    except Exception as exc:
        execution_failure = f"execution error: {type(exc).__name__}"

    if result is not None:
        try:
            get_structured = getattr(agent.profile_store, "get_structured", None)
            if get_structured:
                profile_keys = sorted(get_structured(conv_id))
        except Exception:
            profile_inspection_failure = True

    try:
        agent.clear_conversation_state(conv_id)
    except Exception:
        cleanup_failure = True

    latency_ms = round((time.perf_counter() - started_at) * 1000)
    if result is None:
        failures = [execution_failure or "execution failed"]
        failure_codes = ["EXECUTION_ERROR"]
        if cleanup_failure:
            failures.append("evaluation state cleanup failed")
            failure_codes.append("CLEANUP_ERROR")
        return CaseResult(
            case_id=case.id,
            passed=False,
            stage="",
            tools=[],
            stop_reason="",
            retrieval_triggered=False,
            profile_keys=profile_keys,
            answer_nonempty=False,
            latency_ms=latency_ms,
            failures=failures,
            failure_codes=failure_codes,
        )

    stage = result.get("stage", "")
    tools = list(result.get("executed_tools", []))
    stop_reason = result.get("stop_reason", "")
    retrieval_triggered = bool(result.get("retrieval_triggered", False))
    answer_nonempty = bool(str(result.get("answer", "")).strip())
    failures: list[str] = []
    failure_codes: list[str] = []
    if stage not in case.expected_stages:
        _failure(
            failures,
            failure_codes,
            "STAGE_MISMATCH",
            f"stage={stage!r}, expected one of {case.expected_stages}",
        )
    missing_tools = sorted(set(case.required_tools) - set(tools))
    if missing_tools:
        _failure(
            failures,
            failure_codes,
            "REQUIRED_TOOL_MISSING",
            f"missing required tools: {missing_tools}",
        )
    forbidden_tools = sorted(set(case.forbidden_tools) & set(tools))
    if forbidden_tools:
        _failure(
            failures,
            failure_codes,
            "FORBIDDEN_TOOL_CALLED",
            f"called forbidden tools: {forbidden_tools}",
        )
    if stop_reason not in case.expected_stop_reasons:
        _failure(
            failures,
            failure_codes,
            "STOP_REASON_MISMATCH",
            f"stop_reason={stop_reason!r}, expected one of {case.expected_stop_reasons}"
        )
    if retrieval_triggered != case.expected_retrieval:
        _failure(
            failures,
            failure_codes,
            "RETRIEVAL_TRIGGER_MISMATCH",
            f"retrieval_triggered={retrieval_triggered}, expected {case.expected_retrieval}"
        )
    missing_profile_keys = sorted(set(case.expected_profile_keys) - set(profile_keys))
    if missing_profile_keys:
        _failure(
            failures,
            failure_codes,
            "PROFILE_KEY_MISSING",
            f"missing profile keys: {missing_profile_keys}",
        )
    if not answer_nonempty:
        _failure(failures, failure_codes, "EMPTY_ANSWER", "final answer is empty")
    if profile_inspection_failure:
        _failure(
            failures,
            failure_codes,
            "PROFILE_INSPECTION_ERROR",
            "profile inspection failed",
        )
    if cleanup_failure:
        _failure(
            failures,
            failure_codes,
            "CLEANUP_ERROR",
            "evaluation state cleanup failed",
        )

    return CaseResult(
        case_id=case.id,
        passed=not failures,
        stage=stage,
        tools=tools,
        stop_reason=stop_reason,
        retrieval_triggered=retrieval_triggered,
        profile_keys=profile_keys,
        answer_nonempty=answer_nonempty,
        latency_ms=result.get("latency_ms", latency_ms),
        failures=failures,
        failure_codes=failure_codes,
    )


def _load_live_agent():
    """Validate explicit live prerequisites before loading heavy dependencies."""
    from src.config import config, init_langsmith

    api_key = config.OPENAI_API_KEY.strip()
    if not api_key or api_key in {"sk-placeholder", "your-api-key-here"}:
        raise RuntimeError("OPENAI_API_KEY is not configured for live evaluation")
    if not Path(config.PRODUCT_DB_PATH).is_file():
        raise RuntimeError("PRODUCT_DB_PATH is missing for live evaluation")
    if not Path(config.PRODUCT_CHROMA_DIR).is_dir():
        raise RuntimeError("PRODUCT_CHROMA_DIR is missing for live evaluation")

    init_langsmith()
    from backend.dependencies import get_agent
    return get_agent()


def run_live(
    cases: Sequence[EvalCase], agent=None
) -> tuple[EvalSummary, list[CaseResult]]:
    live_agent = agent or _load_live_agent()
    results = [run_live_case(live_agent, case) for case in cases]
    return summarize(cases, results), results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate shopping Agent behavior")
    parser.add_argument(
        "--mode", choices=("deterministic", "live"), default="deterministic"
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Versioned dataset manifest; auto-selected for the default case file.",
    )
    parser.add_argument(
        "--dataset-id",
        default=None,
        help="Dataset artifact id declared by --manifest.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("evals/results"))
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run only a named case; repeat to select multiple cases.",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit selected cases to control live cost."
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    default_cases = args.cases.resolve() == DEFAULT_CASES_PATH.resolve()
    manifest_path = args.manifest
    if manifest_path is None and default_cases:
        manifest_path = DEFAULT_DATASET_MANIFEST_PATH

    dataset = None
    manifest = None
    dataset_id = args.dataset_id
    if manifest_path is not None:
        if not dataset_id:
            dataset_id = "agent-behavior-v1" if default_cases else None
        if not dataset_id:
            print("--dataset-id is required when --manifest is used with custom cases")
            return 2
        try:
            manifest, dataset = validate_dataset_artifact(
                manifest_path, dataset_id, args.cases
            )
        except ValueError as exc:
            print(f"evaluation manifest unavailable: {exc}")
            return 2

    try:
        cases = load_cases(args.cases)
    except ValueError as exc:
        print(f"evaluation cases unavailable: {exc}")
        return 2
    if args.case_id:
        selected_ids = set(args.case_id)
        cases = [case for case in cases if case.id in selected_ids]
        missing_ids = selected_ids - {case.id for case in cases}
        if missing_ids:
            print(f"unknown case ids: {sorted(missing_ids)}")
            return 2
    if args.limit:
        if args.limit < 1:
            print("--limit must be greater than zero")
            return 2
        cases = cases[: args.limit]
    if not cases:
        print("no evaluation cases selected")
        return 2
    started_at_monotonic, started_at = evaluation_start()
    from backend.logging_config import logger as telemetry_logger
    original_level = telemetry_logger.level
    if not args.verbose:
        telemetry_logger.setLevel(logging.WARNING)
    try:
        if args.mode == "deterministic":
            summary, results = run_deterministic(cases)
        else:
            try:
                summary, results = run_live(cases)
            except RuntimeError as exc:
                print(f"live evaluation unavailable: {exc}")
                return 2
    finally:
        telemetry_logger.setLevel(original_level)

    metrics = asdict(summary)
    gate_spec = (
        get_quality_gate(manifest, dataset_id, args.mode)
        if manifest is not None and dataset_id
        else None
    )
    gate = (
        evaluate_quality_gate(metrics, gate_spec)
        if gate_spec is not None
        else default_zero_failure_gate(metrics, args.mode)
    )
    metadata = build_run_metadata(
        mode=args.mode,
        started_at_monotonic=started_at_monotonic,
        started_at=started_at,
        dataset=dataset,
        cases_path=args.cases,
        model="scripted-eval" if args.mode == "deterministic" else "configured",
        provider="local-double" if args.mode == "deterministic" else "configured",
        external_access=args.mode == "live",
    )
    metadata["dataset"]["selected_case_count"] = len(cases)
    json_path, markdown_path = write_reports(
        summary,
        results,
        args.output_dir,
        mode=args.mode,
        metadata=metadata,
        gate=gate,
    )
    print(
        f"{args.mode}: {summary.passed}/{summary.total} passed "
        f"({summary.pass_rate:.1%}); gate={'passed' if gate.passed else 'failed'}; "
        f"reports: {json_path}, {markdown_path}"
    )
    for result in results:
        if not result.passed:
            print(f"FAIL {result.case_id}: {'; '.join(result.failures)}")
    return 0 if gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
