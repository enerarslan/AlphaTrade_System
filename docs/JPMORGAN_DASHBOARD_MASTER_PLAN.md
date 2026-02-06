# AlphaTrade Institutional Dashboard Master Plan

## 1. Mission
Build a production-grade, institutional control plane that can operate live trading safely, with full observability, risk governance, model transparency, and incident response.

This dashboard is the operator cockpit for:
- `main.py` command workflows
- `scripts/*` operational jobs
- `quant_trading_system/*` runtime state

## 2. Success Criteria (Institutional Baseline)
- Uptime target: 99.9%+ for dashboard API and websocket channels
- Recovery target: < 5 minutes for control-plane restart
- Operator actions (start/stop/kill-switch): < 2s API response p95
- Streaming freshness: < 1s internal event propagation target
- Full auditability of privileged actions
- Role-based access control and hard authorization boundaries
- No silent failures for risk or execution critical paths

## 3. Domain Coverage and Required Modules

### A. Command Center
- Portfolio equity, drawdown, returns, active positions, execution throughput
- Strategy allocation and session state
- Live market regime overlays
- "single pane" summary of risk, orders, alerts, and infra health

### B. Trading Operations
- Start/stop/restart live process with strict guardrails
- Order blotter (status lifecycle, fills, slippage, latency)
- Job orchestration for operational commands (`backtest`, `train`, `health`, `deploy`)
- Circuit-breaker and trading mode indicators (paper/live)

### C. Risk War Room
- VaR/CVaR, drawdown curve, concentration, sector/correlation risk
- Real-time risk limit monitor with breach actions
- Kill-switch controls with dual authorization workflow
- Stress-test drilldown and scenario replay

### D. Model Ops
- Explainability (SHAP/feature importance)
- Drift and staleness monitoring
- Retraining and validation-gate job lifecycle
- Champion/challenger model switch control

### E. Operations Console
- Service health, latency, queue pressure, websocket status
- Structured logs + incident timeline
- Runbook-linked action panel
- Controlled execution of allowlisted maintenance jobs

### F. Alerts and Incident Response
- Alert inbox with severity, ownership, ack/resolve states
- Correlated event graph (risk + infra + execution)
- Escalation policy hooks (email/slack/pager adapters)

### G. Governance and Compliance
- Immutable audit trail for all control actions
- User session analytics and privileged action review
- Retention policy for orders, alerts, logs, and decision metadata

## 4. Current Implementation Snapshot (Completed Foundation)

### Backend (`quant_trading_system/monitoring/dashboard.py`)
- Control plane APIs:
  - `/control/trading/status`
  - `/control/trading/start`
  - `/control/trading/stop`
  - `/control/trading/restart`
  - `/control/risk/kill-switch/activate`
  - `/control/risk/kill-switch/reset`
  - `/control/jobs`
  - `/control/jobs/{job_id}`
  - `/control/jobs/{job_id}/cancel`
- Computed analytics APIs:
  - `/execution/tca`
  - `/risk/var`
  - `/models/explainability`
- Realtime websocket channels:
  - `/ws/portfolio`
  - `/ws/orders`
  - `/ws/signals`
  - `/ws/alerts`
- Event-bus binding and job/task lifecycle tracking in dashboard state

### Frontend (`dashboard_ui/src`)
- Auth-gated shell + route isolation
- Institutional layout with dedicated modules:
  - Command Center
  - Trading Terminal
  - Risk War Room
  - Models
  - Operations
  - Alerts
  - Settings
- Unified store (`zustand`) for:
  - auth session lifecycle
  - snapshot fetchers
  - websocket subscriptions
  - control actions (trading/kill-switch/jobs)

## 5. Integration Plan with `main.py` and `scripts`

### Main command contract
Dashboard uses controlled command orchestration against `main.py`:
- trading lifecycle (`trade` mode)
- operational jobs via allowlisted command map

### Script orchestration
Expose operator-safe wrappers for:
- `scripts/trade.py`
- `scripts/backtest.py`
- `scripts/train.py`
- `scripts/health.py`
- `scripts/deploy.py`
- `scripts/data.py`
- `scripts/features.py`

Rules:
- allowlist only (no arbitrary command execution)
- bounded runtime per job type
- stdout/stderr capture to structured logs
- cancellable job execution with deterministic terminal state

## 6. Architecture Hardening Plan

### Phase 1: Control-Plane Reliability
- Migrate FastAPI startup/shutdown hooks to lifespan handlers
- Add idempotency keys for control actions
- Add persistent state backend (Redis/Postgres) for jobs and session runtime state
- Add action retries with exponential backoff for event publish failures

### Phase 2: Risk and Execution Depth
- Expand risk endpoints for concentration/correlation/stress metrics
- Add execution quality analytics (arrival price delta, venue slippage buckets)
- Add pre-trade and post-trade risk attribution views

### Phase 3: Security and Governance
- RBAC with role-scoped permissions (viewer/operator/risk/admin)
- MFA flow for privileged controls (kill-switch reset, live mode switch)
- Signed audit records and tamper-evident storage

### Phase 4: Model Governance
- Model registry UI integration
- Drift alarms and auto-retraining pipeline triggers
- Validation gates (data quality + statistical significance + risk impact)

### Phase 5: SRE and Incident Operations
- SLO dashboards with burn-rate alerts
- Incident timeline + postmortem exports
- Runbook automation for common failure domains

## 7. Minimum Acceptance Gates per Release
- Unit + integration tests green for dashboard and monitoring flows
- Frontend production build passes
- Manual operator scenarios validated:
  - login/logout
  - trading start/stop/restart
  - kill-switch activate/reset
  - job create/cancel
  - websocket reconnection behavior
- Security checks:
  - endpoint auth verification
  - role restriction tests
  - audit trail assertions for control actions

## 8. Next Execution Backlog (Priority Order)
1. Replace in-memory job/state storage with Redis + persistence fallback.
2. Implement RBAC matrix and protected action middleware.
3. Add idempotency + replay protection for control endpoints.
4. Add structured audit event schema and retention policy.
5. Extend risk endpoints for stress testing and correlation concentration.
6. Add advanced TCA decomposition and execution latency distribution.
7. Add incident timeline UI and on-call escalation hooks.
8. Optimize frontend bundle splitting for faster initial load.

## 9. Implementation Status (2026-02-06)
- Phase 1 reliability baseline implemented:
  - Redis + file fallback persistence for control state
  - Idempotency for privileged control actions
  - Retry with exponential backoff for control event publish
- Phase 2 risk/execution depth implemented:
  - Advanced execution quality endpoint
  - Concentration/correlation/stress/attribution risk endpoints
  - Risk war-room UI integration for advanced metrics
- Phase 3 security/governance baseline implemented:
  - RBAC enforcement and role-aware UI routing/navigation
  - MFA/TOTP checks for privileged actions
  - Signed tamper-evident audit trail + operator UI visibility
- Phase 4 model governance baseline implemented:
  - Model registry/drift/validation/champion-challenger endpoints
  - Champion promotion control endpoint with audit + MFA support
  - Model governance UI with promotion controls and gate visibility
- Phase 5 SRE/incident operations baseline implemented:
  - SLO burn-rate status endpoint
  - Incident list/timeline endpoints
  - Runbook catalog and runbook execution endpoint
  - Operations UI runbook automation + incident timeline panels
