from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Protocol

import polars as pl

from hermes_quant.experience.record import TradeExperience


class ExperienceStore(Protocol):
    """Protocol defining persistence and research export for trade experiences."""

    def save(self, experience: TradeExperience) -> None: ...

    def save_batch(self, experiences: list[TradeExperience]) -> None: ...

    def get(self, trade_id: str) -> TradeExperience | None: ...

    def get_by_decision_id(self, decision_id: str) -> TradeExperience | None: ...

    def list_experiences(
        self,
        *,
        asset: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        outcome_label: str | None = None,
    ) -> list[TradeExperience]: ...

    def to_polars(self) -> pl.DataFrame: ...

    def export_parquet(self, output_path: Path) -> None: ...


class InMemoryExperienceStore:
    """In-memory experience store for testing and fast research iterations."""

    def __init__(self) -> None:
        self._records: dict[str, TradeExperience] = {}
        self._by_decision: dict[str, str] = {}

    def save(self, experience: TradeExperience) -> None:
        self._records[experience.trade_id] = experience
        self._by_decision[experience.decision_id] = experience.trade_id

    def save_batch(self, experiences: list[TradeExperience]) -> None:
        for exp in experiences:
            self.save(exp)

    def get(self, trade_id: str) -> TradeExperience | None:
        return self._records.get(trade_id)

    def get_by_decision_id(self, decision_id: str) -> TradeExperience | None:
        tid = self._by_decision.get(decision_id)
        return self._records.get(tid) if tid else None

    def list_experiences(
        self,
        *,
        asset: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        outcome_label: str | None = None,
    ) -> list[TradeExperience]:
        results: list[TradeExperience] = []
        for exp in self._records.values():
            if asset and exp.asset != asset:
                continue
            if outcome_label and exp.outcome_label != outcome_label:
                continue
            if start_time and exp.decision_time < start_time:
                continue
            if end_time and exp.decision_time > end_time:
                continue
            results.append(exp)
        return sorted(results, key=lambda x: x.decision_time)

    def to_polars(self) -> pl.DataFrame:
        records = [exp.to_dict() for exp in self._records.values()]
        if not records:
            return pl.DataFrame()
        return pl.DataFrame(records).sort("decision_time")

    def export_parquet(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_polars().write_parquet(output_path)


class SQLiteExperienceStore(InMemoryExperienceStore):
    """Crash-durable SQLite experience store."""

    def __init__(self, database_path: str) -> None:
        super().__init__()
        self.db_path = database_path
        self.connection = sqlite3.connect(database_path, check_same_thread=False)
        self._init_tables()
        self._load_existing()

    def _init_tables(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_experience (
                trade_id TEXT PRIMARY KEY,
                decision_id TEXT NOT NULL UNIQUE,
                decision_time TEXT NOT NULL,
                asset TEXT NOT NULL,
                market_state_version TEXT NOT NULL,
                feature_version TEXT NOT NULL,
                model_versions TEXT NOT NULL,
                hermes_model TEXT NOT NULL,
                hermes_prompt_version TEXT NOT NULL,
                action TEXT NOT NULL,
                confidence REAL NOT NULL,
                risk_decision TEXT NOT NULL,
                approved_quantity REAL NOT NULL,
                entry_price REAL,
                exit_price REAL,
                fees REAL NOT NULL,
                slippage REAL NOT NULL,
                realized_pnl REAL,
                net_return REAL,
                holding_period_bars INTEGER,
                outcome_label TEXT NOT NULL,
                regime_at_decision TEXT,
                post_trade_notes TEXT,
                recorded_at TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def _load_existing(self) -> None:
        cur = self.connection.execute("SELECT * FROM trade_experience")
        col_names = [d[0] for d in cur.description]
        for row in cur.fetchall():
            d = dict(zip(col_names, row, strict=False))
            exp = TradeExperience.from_dict(d)
            super().save(exp)

    def save(self, experience: TradeExperience) -> None:
        super().save(experience)
        d = experience.to_dict()
        cols = list(d.keys())
        placeholders = ", ".join(["?"] * len(cols))
        sql = f"""
            INSERT INTO trade_experience ({", ".join(cols)})
            VALUES ({placeholders})
            ON CONFLICT(trade_id) DO UPDATE SET
                exit_price=excluded.exit_price,
                fees=excluded.fees,
                slippage=excluded.slippage,
                realized_pnl=excluded.realized_pnl,
                net_return=excluded.net_return,
                holding_period_bars=excluded.holding_period_bars,
                outcome_label=excluded.outcome_label,
                post_trade_notes=excluded.post_trade_notes;
        """
        self.connection.execute(sql, [d[c] for c in cols])
        self.connection.commit()


class PostgresExperienceStore(InMemoryExperienceStore):
    """Production TimescaleDB/PostgreSQL experience database adapter."""

    def __init__(
        self, dsn: str = "postgresql://hermes:hermes_local_only@localhost:5432/hermes_quant"
    ) -> None:
        super().__init__()
        self.dsn = dsn
        self._init_tables()
        self._load_existing()

    def _init_tables(self) -> None:
        import psycopg

        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS trade_experience (
                        trade_id TEXT PRIMARY KEY,
                        decision_id TEXT NOT NULL UNIQUE,
                        decision_time TIMESTAMPTZ NOT NULL,
                        asset TEXT NOT NULL,
                        market_state_version TEXT NOT NULL,
                        feature_version TEXT NOT NULL,
                        model_versions JSONB NOT NULL,
                        hermes_model TEXT NOT NULL,
                        hermes_prompt_version TEXT NOT NULL,
                        action TEXT NOT NULL,
                        confidence DOUBLE PRECISION NOT NULL,
                        risk_decision TEXT NOT NULL,
                        approved_quantity DOUBLE PRECISION NOT NULL,
                        entry_price DOUBLE PRECISION,
                        exit_price DOUBLE PRECISION,
                        fees DOUBLE PRECISION NOT NULL,
                        slippage DOUBLE PRECISION NOT NULL,
                        realized_pnl DOUBLE PRECISION,
                        net_return DOUBLE PRECISION,
                        holding_period_bars INTEGER,
                        outcome_label TEXT NOT NULL,
                        regime_at_decision TEXT,
                        post_trade_notes TEXT,
                        recorded_at TIMESTAMPTZ NOT NULL
                    );
                    """
                )
            conn.commit()

    def _load_existing(self) -> None:
        import psycopg

        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM trade_experience")
                desc = [d[0] for d in cur.description]
                for row in cur.fetchall():
                    d = dict(zip(desc, row, strict=False))
                    # Handle jsonb/datetime conversion
                    if isinstance(d.get("model_versions"), str):
                        try:
                            d["model_versions"] = json.loads(d["model_versions"])
                        except Exception:
                            d["model_versions"] = {}
                    exp = TradeExperience.from_dict(d)
                    super().save(exp)

    def save(self, experience: TradeExperience) -> None:
        super().save(experience)
        import psycopg

        d = experience.to_dict()
        cols = list(d.keys())
        placeholders = ", ".join(["%s"] * len(cols))
        sql = f"""
            INSERT INTO trade_experience ({", ".join(cols)})
            VALUES ({placeholders})
            ON CONFLICT (trade_id) DO UPDATE SET
                exit_price=EXCLUDED.exit_price,
                fees=EXCLUDED.fees,
                slippage=EXCLUDED.slippage,
                realized_pnl=EXCLUDED.realized_pnl,
                net_return=EXCLUDED.net_return,
                holding_period_bars=EXCLUDED.holding_period_bars,
                outcome_label=EXCLUDED.outcome_label,
                post_trade_notes=EXCLUDED.post_trade_notes;
        """
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, [d[c] for c in cols])
            conn.commit()
