# Technical Debt Register

This is the canonical register for deliberate scope deferrals and known safety gaps. Every deliberate deferral must be recorded here when the decision is made, including decisions made during Phases 4 through 7. An item is closed only when its explicit resolution condition is met and verified by tests or equivalent evidence.

## TD-001: Dedup key includes exact market data

- **ID:** TD-001
- **Title:** Dedup key includes exact market data (price/volatility/volume), not just decision identity
- **Description:** The Phase 7 orchestrator deduplication key hashes the validated Hermes decision together with exact price, estimated volatility, and average daily volume inputs.
- **Impact:** A retry with even slightly different market inputs mints a new `decision_id` and bypasses orchestrator-level deduplication.
- **Current workaround:** Rely on broker-level `decision_id` idempotency and maintain operational discipline to avoid blind retries.
- **Owner:** Trading/orchestration maintainers
- **Status:** Open - V1 limitation
- **Resolution condition:** Implement trading-session/date-based identity once market-session scheduling exists, then add tests proving retries within the same session deduplicate despite changed live market inputs.

## TD-002: Unexpected position state reconciliation is not implemented

- **ID:** TD-002
- **Title:** `unexpected_position_state` halt trigger is an untested/unimplemented reconciliation placeholder
- **Description:** The orchestrator has a structural halt flag for unexpected position state, but no production reconciliation currently computes order/fill consistency against expected portfolio state.
- **Impact:** The halt trigger exists structurally but nothing computes real position/order consistency, so this safety layer is currently a no-op in practice.
- **Current workaround:** None. This is a known gap in automatic safety coverage, not merely an inconvenience.
- **Owner:** Trading/risk maintainers
- **Status:** Open - deferred safety work
- **Resolution condition:** Implement reconciliation logic comparing persisted orders and fills against expected portfolio state, wire it to the halt trigger, and add dedicated tests covering consistent state, unexpected state, and crash/restart behavior.
