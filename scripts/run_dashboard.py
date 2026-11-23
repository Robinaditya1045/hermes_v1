#!/usr/bin/env python3
"""Run the Hermes Quant Observability Dashboard & API (Phase 9)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root and src to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

import uvicorn  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Hermes Quant Observability Dashboard & REST API"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host interface to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable hot reloading for development",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Hermes Quant — Observability & Trading Dashboard (Phase 9)")
    print(f"Listening on http://{args.host}:{args.port}")
    print(f"Web UI:       http://localhost:{args.port}/")
    print(f"REST Docs:    http://localhost:{args.port}/docs")
    print(f"Audit API:    http://localhost:{args.port}/api/v1/audit/{{trade_id}}")
    print("=" * 70)

    uvicorn.run(
        "apps.dashboard.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
