# Hermes Quant

Phases 0 through 4 of the research-first paper-trading platform described in the system design document.

## Scope

This slice contains the repository foundation, fail-closed OpenBB daily OHLCV ingestion, technical features, a walk-forward return model, regime/risk models, and deterministic risk controls. It does not include Hermes tool integration, paper execution, or learning.

The ingestion contract is:

`information_available_at <= decision_at < outcome_at`

Provider selection is configurable through `OpenBBDataAdapter(provider=...)`; no provider credentials are committed to the repository.

## Local checks

```powershell
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
```

Start the local TimescaleDB service with `docker compose up -d postgres`.

## Kill switch status

The deterministic risk engine accepts `kill_switch_active` as explicit portfolio state and fails closed when it is true. Automatic triggering and persistence are deferred until the portfolio-state and observability workflows are implemented. The future trigger sources are the daily loss limit, repeated Phase 1 data-validation failures, and a configured maximum-drawdown threshold.
