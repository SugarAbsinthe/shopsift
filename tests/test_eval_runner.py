"""Deterministic Agent evaluation runner tests."""

import json

from evals.runner import (
    main,
    run_deterministic,
    run_deterministic_case,
    run_live,
    write_reports,
)
from evals.schema import EvalCase, load_cases


def test_default_deterministic_suite_passes():
    cases = load_cases()
    summary, results = run_deterministic(cases)
    assert summary.total == 25
    assert summary.passed == 25
    assert summary.pass_rate == 1.0
    assert summary.stage_accuracy == 1.0
    assert summary.tool_boundary_accuracy == 1.0
    assert summary.stop_reason_accuracy == 1.0
    assert summary.retrieval_accuracy == 1.0
    assert summary.nonempty_answer_rate == 1.0
    assert summary.illegal_tool_calls == 0
    assert all(result.passed for result in results)


def test_case_failure_contains_objective_reasons():
    case = EvalCase.model_validate({
        "id": "expected_failure",
        "question": "你好",
        "expected_stages": ["search"],
        "expected_retrieval": True,
    })
    result = run_deterministic_case(case)
    assert not result.passed
    assert any("stage=" in failure for failure in result.failures)
    assert any("retrieval_triggered=" in failure for failure in result.failures)
    assert "STAGE_MISMATCH" in result.failure_codes
    assert "RETRIEVAL_TRIGGER_MISMATCH" in result.failure_codes


def test_case_execution_error_isolated_as_a_failure(monkeypatch):
    import evals.runner as runner

    class BrokenGraph:
        def __init__(self, **kwargs):
            raise RuntimeError("secret fixture detail")

    monkeypatch.setattr(runner, "ShoppingGuideGraph", BrokenGraph)
    case = EvalCase.model_validate({
        "id": "isolated_error",
        "question": "你好",
        "expected_stages": ["discovery"],
        "expected_retrieval": False,
    })
    result = run_deterministic_case(case)
    assert result.passed is False
    assert result.failure_codes == ["EXECUTION_ERROR"]
    assert result.failures == ["execution error: RuntimeError"]
    assert "secret" not in str(result.failures)


def test_tool_error_and_loop_limit_cases_pass():
    selected = {
        case.id: case
        for case in load_cases()
        if case.id in {"tool_error_fallback", "max_tool_rounds"}
    }
    tool_error = run_deterministic_case(selected["tool_error_fallback"])
    loop_limit = run_deterministic_case(selected["max_tool_rounds"])
    assert tool_error.passed
    assert tool_error.stop_reason == "tool_error"
    assert loop_limit.passed
    assert loop_limit.stop_reason == "max_tool_rounds"
    assert loop_limit.tools == ["get_product_detail"]


def test_report_writer_outputs_json_and_markdown(tmp_path):
    cases = load_cases()[:2]
    summary, results = run_deterministic(cases)
    json_path, markdown_path = write_reports(summary, results, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "deterministic"
    assert payload["report_schema_version"] == 1
    assert payload["summary"]["total"] == 2
    assert payload["summary"]["failure_counts"] == {}
    assert "Pass rate" in markdown_path.read_text(encoding="utf-8")


def test_default_cli_writes_manifest_and_gate_metadata(tmp_path):
    exit_code = main([
        "--output-dir", str(tmp_path),
    ])
    assert exit_code == 0
    payload = json.loads((tmp_path / "deterministic.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["dataset"]["dataset_id"] == "agent-behavior-v1"
    assert payload["gate"]["name"] == "agent-deterministic-regression"
    assert payload["gate"]["passed"] is True


def test_cli_returns_nonzero_for_failed_suite(tmp_path):
    cases_path = tmp_path / "failed.jsonl"
    cases_path.write_text(
        json.dumps({
            "id": "cli_failure",
            "question": "你好",
            "expected_stages": ["search"],
            "expected_retrieval": True,
        }, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    exit_code = main([
        "--cases", str(cases_path),
        "--output-dir", str(tmp_path / "reports"),
    ])
    assert exit_code == 1


class FakeLiveProfileStore:
    def get_structured(self, conv_id):
        return {"budget": {"value": "6000"}}


class FakeLiveAgent:
    def __init__(self):
        self.profile_store = FakeLiveProfileStore()
        self.cleared = []

    def run(self, question, conv_id, chat_history):
        return {
            "answer": "已记录预算。",
            "stage": "needs_elicitation",
            "executed_tools": ["update_user_profile"],
            "stop_reason": "completed",
            "retrieval_triggered": False,
            "latency_ms": 42,
        }

    def clear_conversation_state(self, conv_id):
        self.cleared.append(conv_id)


def test_live_mode_uses_shared_expectations_and_cleans_state():
    case = EvalCase.model_validate({
        "id": "live_profile",
        "question": "预算6000元，请记住",
        "expected_stages": ["needs_elicitation"],
        "required_tools": ["update_user_profile"],
        "expected_retrieval": False,
        "expected_profile_keys": ["budget"],
        "scripted_tool_rounds": [["update_user_profile"]],
    })
    agent = FakeLiveAgent()
    summary, results = run_live([case], agent=agent)
    assert summary.passed == 1
    assert results[0].latency_ms == 42
    assert len(agent.cleared) == 1
    assert agent.cleared[0].startswith("eval-live-live_profile-")


def test_live_mode_records_exception_type_without_detail():
    class FailingAgent(FakeLiveAgent):
        def run(self, question, conv_id, chat_history):
            raise ValueError("secret provider response")

    case = EvalCase.model_validate({
        "id": "live_failure",
        "question": "你好",
        "expected_stages": ["discovery"],
        "expected_retrieval": False,
    })
    _, results = run_live([case], agent=FailingAgent())
    assert results[0].failures == ["execution error: ValueError"]
    assert "secret" not in str(results[0].failures)


def test_live_mode_marks_profile_inspection_failure_and_still_cleans():
    class BrokenProfileStore:
        def get_structured(self, conv_id):
            raise RuntimeError("database detail")

    agent = FakeLiveAgent()
    agent.profile_store = BrokenProfileStore()
    case = EvalCase.model_validate({
        "id": "profile_inspection_failure",
        "question": "预算6000元，请记住",
        "expected_stages": ["needs_elicitation"],
        "required_tools": ["update_user_profile"],
        "expected_retrieval": False,
        "scripted_tool_rounds": [["update_user_profile"]],
    })
    _, results = run_live([case], agent=agent)
    assert "profile inspection failed" in results[0].failures
    assert len(agent.cleared) == 1


def test_live_cli_preflight_failure_is_safe(monkeypatch, tmp_path):
    import evals.runner as runner

    monkeypatch.setattr(
        runner,
        "_load_live_agent",
        lambda: (_ for _ in ()).throw(RuntimeError("OPENAI_API_KEY is not configured")),
    )
    exit_code = main([
        "--mode", "live",
        "--limit", "1",
        "--output-dir", str(tmp_path),
    ])
    assert exit_code == 2
