import argparse
from pathlib import Path

import requests

from hermes_quant.hermes_validation.harness import (
    HarnessRun,
    ObservationHarness,
    ScriptedHermesRuntime,
    scenarios,
    write_observations,
)
from hermes_quant.hermes_validation.llm_runtime import LLMHermesRuntime


def is_endpoint_reachable(base_url: str, timeout: float = 2.0) -> bool:
    try:
        url = f"{base_url.rstrip('/')}/models"
        resp = requests.get(url, timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def print_behavioral_report(runs: tuple[HarnessRun, ...]) -> None:
    print("\n" + "=" * 80)
    print("           HERMES AGENT BEHAVIORAL VALIDATION REPORT (PHASE 7.5)")
    print("=" * 80)

    for run in runs:
        print(f"\n--- Scenario: {run.scenario} (Asset: {run.market_state.get('asset')}) ---")
        dec = run.final_decision or {}
        action = dec.get("action", "NONE")
        confidence = dec.get("confidence", 0.0)
        notional = dec.get("requested_notional", 0.0)
        print(
            f"  Decision:           {action} "
            f"(Confidence: {confidence:.2f}, Advisory Notional: ${notional:,.2f})"
        )
        print(
            f"  First-Attempt Valid: {run.decision_validated_first_attempt} "
            f"(Repairs: {run.decision_repair_attempts})"
        )
        print(f"  Tool Calls Made:    {len(run.tool_calls)}")

        # Tool audit
        invalid_args = [tc for tc in run.tool_calls if tc.invalid_arguments]
        errored_calls = [tc for tc in run.tool_calls if tc.status == "error"]
        if invalid_args:
            print(
                f"  [!] INVALID ARGS:   {len(invalid_args)} call(s) passed invalid argument schema:"
            )
            for tc in invalid_args:
                print(f"      - {tc.tool}({tc.arguments}) -> {tc.response}")
        if errored_calls:
            print(f"  [*] Errored Calls:  {len(errored_calls)} call(s) encountered errors:")
            for tc in errored_calls:
                print(f"      - {tc.tool}({tc.arguments}) -> {tc.response}")

        # Hallucination / Evidence Audit
        evidence_used = dec.get("evidence_used", [])
        print(f"  Evidence Claimed:   {evidence_used}")
        fundamentals = run.market_state.get("fundamentals", {})
        news = run.market_state.get("news", {})

        hallucinations: list[str] = []
        if not fundamentals.get("available") and "fundamentals" in evidence_used:
            hallucinations.append(
                "Claimed 'fundamentals' evidence when fundamentals.available was False"
            )
        if not news.get("available") and "news" in evidence_used:
            hallucinations.append("Claimed 'news' evidence when news.available was False")
        if not news.get("available") and "recent_news" in evidence_used:
            hallucinations.append("Claimed 'recent_news' evidence when news.available was False")

        if hallucinations:
            print(f"  [!] HALLUCINATIONS DETECTED ({len(hallucinations)}):")
            for h in hallucinations:
                print(f"      - {h}")
        else:
            print("  Evidence Integrity: PASS (No evidence hallucinations detected)")

        # Thesis & Risks
        thesis = dec.get("thesis", [])
        risks = dec.get("risks", [])
        print(f"  Thesis ({len(thesis)} items):")
        for t in thesis:
            print(f"    + {t}")
        print(f"  Identified Risks ({len(risks)} items):")
        for r in risks:
            print(f"    - {r}")

        # Metrics
        print(f"  Latency:            {run.total_latency_ms:.1f}ms")
        if run.token_usage:
            print(f"  Token Usage:        {run.token_usage}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total Scenarios Evaluated: {len(runs)}")
    print(f"  Real Runtime Available:    {any(r.runtime_available for r in runs)}")
    print(f"  All First-Attempt Valid:   {all(r.decision_validated_first_attempt for r in runs)}")
    print(f"  Total Repairs Required:    {sum(r.decision_repair_attempts for r in runs)}")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hermes Agent Phase 7.5 Validation Harness")
    parser.add_argument(
        "--runtime",
        choices=["auto", "llm", "scripted"],
        default="auto",
        help="Runtime: 'llm' for real LLM, 'scripted' for mock observer, 'auto' to auto-detect",
    )
    parser.add_argument(
        "--base-url", default=None, help="Base URL for OpenAI-compatible LLM endpoint"
    )
    parser.add_argument("--model", default=None, help="Model name (e.g. llama3.2:1b, hermes3)")
    parser.add_argument("--api-key", default=None, help="API key for LLM endpoint")
    parser.add_argument(
        "--output",
        default="artifacts/hermes_validation/observation.jsonl",
        help="Path to output observation JSONL file",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Timeout in seconds for LLM HTTP requests (default: 180.0)",
    )
    args = parser.parse_args()

    base_url = args.base_url or "http://127.0.0.1:11434/v1"
    if args.runtime == "llm":
        runtime = LLMHermesRuntime(
            base_url=base_url,
            model=args.model,
            api_key=args.api_key,
            timeout_seconds=args.timeout,
        )
    elif args.runtime == "scripted":
        runtime = ScriptedHermesRuntime()
    else:  # auto
        if is_endpoint_reachable(base_url):
            print(f"Detected reachable LLM endpoint at {base_url}. Using real LLMHermesRuntime.")
            runtime = LLMHermesRuntime(
                base_url=base_url,
                model=args.model,
                api_key=args.api_key,
                timeout_seconds=args.timeout,
            )
        else:
            print(
                f"LLM endpoint at {base_url} not reachable. Falling back to ScriptedHermesRuntime."
            )
            runtime = ScriptedHermesRuntime()

    harness = ObservationHarness(runtime=runtime)
    runs = tuple(harness.run(scenario) for scenario in scenarios())
    output = Path(args.output)
    write_observations(output, runs)
    print(f"Wrote {len(runs)} observation runs to {output}")
    print("runtime_available:", any(run.runtime_available for run in runs))
    print_behavioral_report(runs)


if __name__ == "__main__":
    main()
