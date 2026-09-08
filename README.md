# Hermes Quant

Phase 0 and Phase 1 of the research-first paper-trading platform described in the system design document.

## Scope

This slice contains the repository foundation and a fail-closed OpenBB daily OHLCV ingestion path. It does not include features, models, Hermes, risk, paper execution, or learning.

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
