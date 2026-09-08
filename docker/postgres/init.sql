CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS ingestion_batches (
    batch_id UUID PRIMARY KEY,
    provider TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    request_parameters JSONB NOT NULL,
    retrieved_at TIMESTAMPTZ NOT NULL,
    schema_version TEXT NOT NULL,
    row_count INTEGER NOT NULL CHECK (row_count >= 0),
    source_metadata JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS market_bars (
    information_available_at TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    bar_date DATE NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    batch_id UUID NOT NULL REFERENCES ingestion_batches(batch_id),
    PRIMARY KEY (information_available_at, symbol)
);

SELECT create_hypertable('market_bars', 'information_available_at', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS orders (
    order_id UUID PRIMARY KEY,
    decision_id TEXT NOT NULL UNIQUE,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,
    quantity DOUBLE PRECISION NOT NULL,
    notional DOUBLE PRECISION NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS order_transitions (
    transition_id BIGSERIAL PRIMARY KEY,
    order_id UUID NOT NULL REFERENCES orders(order_id),
    from_status TEXT,
    to_status TEXT NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS fills (
    fill_id UUID PRIMARY KEY,
    order_id UUID NOT NULL REFERENCES orders(order_id),
    quantity DOUBLE PRECISION NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    fees DOUBLE PRECISION NOT NULL,
    slippage DOUBLE PRECISION NOT NULL,
    fill_model_version TEXT NOT NULL,
    fill_model_config JSONB NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL
);
