from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_quant.experience import (
    ControlledLearningEngine,
    ExperienceAnalyzer,
    PostgresExperienceStore,
    SQLiteExperienceStore,
)


def format_report_display(report_dict: dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("        HERMES QUANT — SECTION 17 EXPERIENCE & LEARNING REPORT")
    print("=" * 80)
    print(f"  Total Trades Recorded:    {report_dict['total_trades']}")
    print(f"  Closed / Resolved Trades: {report_dict['closed_trades']}")
    print(f"  Open Trades:              {report_dict['open_trades']}")
    print(f"  Risk-Rejected Trades:     {report_dict['risk_rejected_trades']}")
    print(f"  Win Rate (Closed):        {report_dict['win_rate'] * 100:.2f}%")
    print(f"  Profit Factor:            {report_dict['profit_factor']:.2f}")
    print(f"  Total Realized PnL:       ${report_dict['total_realized_pnl']:,.2f}")
    print(f"  Average Trade Return:     {report_dict['average_trade_return'] * 100:+.2f}%")

    # Calibration Table
    print("\n" + "-" * 80)
    print("  CONFIDENCE CALIBRATION ANALYSIS (Expected vs Realized)")
    print("-" * 80)
    print(f"  {'Confidence Bin':<24} | {'Trades':<7} | {'Win Rate':<9} | {'Avg Return':<10}")
    print("  " + "-" * 76)
    for b in report_dict["calibration_bins"]:
        print(
            f"  {b['bin_label']:<24} | "
            f"{b['trade_count']:<7} | "
            f"{b['win_rate'] * 100:>7.2f}% | "
            f"{b['average_return'] * 100:>9.2f}%"
        )

    # Regime Breakdown
    if report_dict["regime_breakdown"]:
        print("\n" + "-" * 80)
        print("  PERFORMANCE BY MARKET REGIME")
        print("-" * 80)
        print(f"  {'Regime':<20} | {'Trades':<7} | {'Win Rate':<9} | {'Total PnL':<12}")
        print("  " + "-" * 76)
        for r, metrics in report_dict["regime_breakdown"].items():
            print(
                f"  {r:<20} | "
                f"{int(metrics['count']):<7} | "
                f"{metrics['win_rate'] * 100:>7.2f}% | "
                f"${metrics['total_pnl']:>10,.2f}"
            )

    # Failure Modes
    if report_dict["failure_modes"]:
        print("\n" + "-" * 80)
        print("  DIAGNOSED FAILURE MODES & BLINDSPOTS")
        print("-" * 80)
        for fm in report_dict["failure_modes"]:
            print(
                f"  [!] {fm['name'].upper()} "
                f"(Severity: {fm['severity']}, Samples: {fm['sample_size']})"
            )

            print(f"      Description: {fm['description']}")
            print(f"      Avg Loss:    {fm['average_loss'] * 100:.2f}%")
            print(f"      Guardrail:   {fm['suggested_guardrail']}")
    print("=" * 80 + "\n")


def run_experience_learning(
    *,
    db_type: str = "sqlite",
    db_path: str = "artifacts/experience/experience.db",
    export_parquet: Path | None = Path("artifacts/experience/trade_experience.parquet"),
    proposals_path: Path | None = Path("artifacts/experience/improvement_proposals.json"),
) -> None:
    if db_type == "postgres":
        print("Connecting to Postgres Experience Store...")
        store = PostgresExperienceStore()
    else:
        print(f"Connecting to SQLite Experience Store ({db_path})...")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        store = SQLiteExperienceStore(db_path)

    experiences = store.list_experiences()
    print(f"Retrieved {len(experiences)} trade experience records.")

    analyzer = ExperienceAnalyzer()
    report = analyzer.analyze(experiences)
    format_report_display(report.to_dict())

    # Controlled Learning Loop (Section 17.2)
    engine = ControlledLearningEngine(analyzer)
    candidates = engine.generate_improvement_candidates(experiences)

    print("=" * 80)
    print("        SECTION 17.2 CONTROLLED IMPROVEMENT CANDIDATES")
    print("=" * 80)
    if candidates:
        print(f"  Generated {len(candidates)} candidate improvements for promotion review:")
        for idx, cand in enumerate(candidates, 1):
            print(
                f"\n  [{idx}] Candidate ID: {cand.candidate_id} (Target: {cand.target_component})"
            )
            print(f"      Status:     [{cand.status}] (Promotion Gate: GATED)")
            print(f"      Hypothesis: {cand.hypothesis}")
            print(f"      Rule:       {cand.proposed_rule}")
            print(f"      Evidence:   {len(cand.evidence_trade_ids)} trade record(s)")
    else:
        print("  No new systemic failure patterns detected exceeding threshold.")
    print("=" * 80 + "\n")

    if export_parquet:
        export_parquet.parent.mkdir(parents=True, exist_ok=True)
        store.export_parquet(export_parquet)
        print(f"Exported research dataset to {export_parquet}")

    if proposals_path:
        proposals_path.parent.mkdir(parents=True, exist_ok=True)
        cand_data = [
            {
                "candidate_id": c.candidate_id,
                "target_component": c.target_component,
                "hypothesis": c.hypothesis,
                "proposed_rule": c.proposed_rule,
                "status": c.status,
                "evidence_trade_ids": list(c.evidence_trade_ids),
                "created_at": c.created_at.isoformat(),
            }
            for c in candidates
        ]
        with open(proposals_path, "w") as f:
            json.dump(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "total_candidates": len(candidates),
                    "candidates": cand_data,
                },
                f,
                indent=2,
            )
        print(f"Saved candidate improvement proposals to {proposals_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hermes Quant — Section 17 Experience Analysis & Controlled Learning"
    )
    parser.add_argument(
        "--db-type",
        choices=["sqlite", "postgres"],
        default="sqlite",
        help="Database backend to query experience from",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="artifacts/experience/experience.db",
        help="Path to SQLite experience database",
    )
    parser.add_argument(
        "--export-parquet",
        type=Path,
        default=Path("artifacts/experience/trade_experience.parquet"),
        help="Path to export research parquet dataset",
    )
    parser.add_argument(
        "--proposals-path",
        type=Path,
        default=Path("artifacts/experience/improvement_proposals.json"),
        help="Path to save candidate improvement proposals JSON",
    )

    args = parser.parse_args()
    run_experience_learning(
        db_type=args.db_type,
        db_path=args.db_path,
        export_parquet=args.export_parquet,
        proposals_path=args.proposals_path,
    )


if __name__ == "__main__":
    main()
