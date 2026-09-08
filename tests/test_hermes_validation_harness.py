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
