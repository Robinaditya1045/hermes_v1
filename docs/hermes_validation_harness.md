# Hermes Validation Harness

This is a standalone, manually triggered observation harness. It does not import or call the Phase 7 orchestrator, paper broker, risk engine, or execution code.

Run it with:

```powershell
python scripts/run_hermes_validation_harness.py
```

The run writes one JSON object per scenario to `artifacts/hermes_validation/observation.jsonl`. Each record contains the exact MarketState JSON, available tool names, tool arguments and responses, timestamps, latency, invalid-argument flags, final decision, first-attempt validation status, repair count, runtime availability, token usage, and total latency.

## Data provenance

The repository has no persisted historical Parquet dataset and no Hermes runtime installation. The four MarketState objects are therefore fixed representative fixtures built from the validated Phase 2-4 schema and version identifiers, not claims of live or historical model output. The harness records `runtime_available: false` and uses a deterministic scripted observer until a Hermes runtime is deliberately installed and manually connected.

Scenarios:

- clearly bullish
- bearish/high-risk
- ambiguous/conflicting signals
- missing-symbol tool error

The harness structurally asserts that the Phase 5 read-only tool surface contains no `submit_paper_order`, broker, order-factory, or risk-evaluation entry point. It does not start a scheduler or autonomous loop.