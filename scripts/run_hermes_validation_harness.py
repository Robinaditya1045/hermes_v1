from pathlib import Path

from hermes_quant.hermes_validation.harness import ObservationHarness, scenarios, write_observations


def main() -> None:
    harness = ObservationHarness()
    runs = tuple(harness.run(scenario) for scenario in scenarios())
    output = Path("artifacts/hermes_validation/observation.jsonl")
    write_observations(output, runs)
    print(f"wrote {len(runs)} observation runs to {output}")
    print("runtime_available:", any(run.runtime_available for run in runs))


if __name__ == "__main__":
    main()
