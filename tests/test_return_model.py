from datetime import datetime, timedelta, timezone

import polars as pl

from hermes_quant.features.technical import calculate_technical_features
from hermes_quant.models.return_model import (
    FEATURE_COLUMNS,
    MODEL_VERSION,
    build_labeled_dataset,
    build_walk_forward_splits,
    fit_walk_forward,
)


def market_bars(row_count: int = 420) -> pl.DataFrame:
    dates = [datetime(2025, 1, 1) + timedelta(days=index) for index in range(row_count)]
    closes = [100.0 + index * 0.1 + (index % 7) * 0.03 for index in range(row_count)]
    return pl.DataFrame(
        {
            "symbol": ["AAPL"] * row_count,
            "bar_date": dates,
            "open": [close - 0.5 for close in closes],
            "high": [close + 1.0 for close in closes],
            "low": [close - 1.0 for close in closes],
            "close": closes,
            "volume": [1_000_000.0 + index * 1000 for index in range(row_count)],
            "information_available_at": [
                datetime(2025, 1, 1, 21, 0, tzinfo=timezone.utc) + timedelta(days=index)
                for index in range(row_count)
            ],
        }
    )


def labeled_dataset() -> pl.DataFrame:
    bars = market_bars()
    features = calculate_technical_features(
        bars,
        decision_at=datetime(2027, 1, 1, 20, 0, tzinfo=timezone.utc),
        outcome_at=datetime(2027, 1, 2, 20, 0, tzinfo=timezone.utc),
    )
    return build_labeled_dataset(features, bars)


def test_walk_forward_dates_are_strictly_disjoint_and_ordered() -> None:
    dataset = labeled_dataset()
    folds = build_walk_forward_splits(
        dataset, train_periods=120, validation_periods=30, test_periods=30
    )

    assert folds
    for fold in folds:
        assert max(fold.train_dates) < min(fold.validation_dates)
        assert max(fold.validation_dates) < min(fold.test_dates)
        assert set(fold.train_dates).isdisjoint(fold.validation_dates)
        assert set(fold.train_dates).isdisjoint(fold.test_dates)
        assert set(fold.validation_dates).isdisjoint(fold.test_dates)


def test_model_uses_only_technical_features_and_emits_market_state_prediction() -> None:
    dataset = labeled_dataset()
    result = fit_walk_forward(dataset, train_periods=120, validation_periods=30, test_periods=30)

    assert set(FEATURE_COLUMNS).issubset(dataset.columns)
    assert set(result.predictions.columns) >= {
        "asset",
        "decision_time",
        "horizon",
        "probability_up",
        "expected_return",
        "model_version",
    }
    assert result.predictions.get_column("model_version").unique().to_list() == [MODEL_VERSION]
    assert result.predictions.get_column("probability_up").is_between(0.0, 1.0).all()
    assert set(result.metrics) == {
        "model_cumulative_return",
        "buy_and_hold_cumulative_return",
    }
