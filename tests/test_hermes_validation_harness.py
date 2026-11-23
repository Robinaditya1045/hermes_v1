import json

from hermes_quant.hermes_validation.harness import (
    FORBIDDEN_TOOLS,
    READ_ONLY_TOOLS,
    ObservationHarness,
    scenarios,
    write_observations,
)


def test_harness_runs_all_observation_scenarios_without_execution_surface() -> None:
    harness = ObservationHarness()
    runs = [harness.run(scenario) for scenario in scenarios()]

    assert len(runs) == 4
    assert all(run.available_tools == READ_ONLY_TOOLS for run in runs)
    assert all(not set(run.available_tools) & FORBIDDEN_TOOLS for run in runs)
    assert all(run.final_decision for run in runs)
    assert all(run.decision_validated_first_attempt for run in runs)
    assert any(call.status == "error" for run in runs for call in run.tool_calls)
    assert all(run.runtime_available is False for run in runs)


def test_harness_persists_one_json_line_per_run(tmp_path) -> None:
    runs = tuple(ObservationHarness().run(scenario) for scenario in scenarios())
    output = tmp_path / "observations.jsonl"

    write_observations(output, runs)

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 4
    assert all(record["market_state"]["schema_version"] == "market_state.v1" for record in records)
    assert all("tool_calls" in record for record in records)
    assert all("available_tools" in record for record in records)


def test_llm_runtime_protocol_compliance() -> None:
    from hermes_quant.hermes_validation.llm_runtime import LLMHermesRuntime

    runtime = LLMHermesRuntime()
    assert runtime.runtime_available is True
    assert runtime.token_usage is None


def test_llm_runtime_decide_with_mocked_llm(monkeypatch) -> None:
    from hermes_quant.hermes_validation.harness import ObservationHarness, scenarios
    from hermes_quant.hermes_validation.llm_runtime import LLMHermesRuntime

    scenario = scenarios()[0]  # AAPL clearly_bullish

    call_count = 0

    class MockResponse:
        def __init__(self, json_data, status_code=200):
            self._json_data = json_data
            self.status_code = status_code

        def json(self):
            return self._json_data

        def raise_for_status(self):
            pass

    def mock_post(url, headers=None, json=None, timeout=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First turn: LLM calls get_market_regime
            return MockResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_123",
                                        "type": "function",
                                        "function": {
                                            "name": "get_market_regime",
                                            "arguments": '{"symbol": "AAPL"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                }
            )
        else:
            # Second turn: LLM provides final decision JSON
            decision_content = """```json
{
  "asset": "AAPL",
  "action": "BUY",
  "confidence": 0.85,
  "thesis": ["Bullish trend confirmed", "Model predicted positive return"],
  "risks": ["Drawdown risk within tolerance"],
  "evidence_used": ["market_state", "market_regime"],
  "requested_notional": 1500.0
}
```"""
            return MockResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": decision_content,
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 150, "completion_tokens": 50, "total_tokens": 200},
                }
            )

    import requests

    monkeypatch.setattr(requests, "post", mock_post)

    runtime = LLMHermesRuntime(base_url="http://mock-llm/v1", model="test-model")
    harness = ObservationHarness(runtime=runtime)
    run = harness.run(scenario)

    assert run.runtime_available is True
    assert run.decision_validated_first_attempt is True
    assert run.decision_repair_attempts == 0
    assert run.final_decision is not None
    assert run.final_decision["action"] == "BUY"
    assert run.final_decision["confidence"] == 0.85
    assert len(run.tool_calls) == 1
    assert run.tool_calls[0].tool == "get_market_regime"
    assert run.tool_calls[0].status == "success"
    assert run.token_usage == {"prompt_tokens": 250, "completion_tokens": 70, "total_tokens": 320}
