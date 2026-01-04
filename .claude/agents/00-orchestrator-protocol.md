# Master Orchestrator Protocol v2.0

## Purpose
Coordinates all specialized agents for AlphaTrade System. This is the **brain** of the multi-agent system.

## Agent Registry

| ID | Agent | Alias | Priority | Scope |
|----|-------|-------|----------|-------|
| 01 | System Architect | `@architect` | Highest | Core, config, interfaces |
| 02 | ML/Quant Engineer | `@mlquant` | High | Models, features, alphas |
| 03 | Trading & Execution | `@trader` | **CRITICAL** | Execution, risk, orders |
| 04 | Data Engineer | `@data` | High | Pipelines, DB, storage |
| 05 | Infrastructure & QA | `@infra` | High | Docker, tests, monitoring |
| 06 | Code Hygiene | `@hygiene` | High | Dead code, cleanup |
| 07 | Semantic Validator | `@validator` | **CRITICAL** | Logic errors, safety |

## Priority Resolution

When agents conflict, resolve in this order:
```
1. @validator  (CRITICAL) - Logic errors = money lost
2. @trader     (CRITICAL) - Safety/risk override everything
3. @architect  (Highest)  - Architecture decisions
4. @hygiene    (High)     - Code quality
5. @mlquant    (High)     - ML/Quant specifics
6. @data       (High)     - Data pipeline
7. @infra      (High)     - Infrastructure
```

## Decision Matrix

### Quick Reference: Which Agent for What?

```
┌─────────────────────────────────────────────────────────────────────────┐
│ TASK TYPE                        │ PRIMARY AGENT  │ SUPPORT AGENTS      │
├─────────────────────────────────────────────────────────────────────────┤
│ Bug fix in execution/order code  │ @trader        │ @validator          │
│ Bug fix in model/feature code    │ @mlquant       │ @validator          │
│ Bug fix in data pipeline         │ @data          │ @validator          │
│ Bug fix elsewhere                │ Domain-specific│ @hygiene            │
├─────────────────────────────────────────────────────────────────────────┤
│ New trading strategy             │ @architect     │ @mlquant → @trader  │
│ New ML model                     │ @mlquant       │ @infra (tests)      │
│ New data source                  │ @data          │ @architect          │
│ New API endpoint                 │ @architect     │ @infra              │
├─────────────────────────────────────────────────────────────────────────┤
│ Performance issue                │ @infra         │ Domain-specific     │
│ Security concern                 │ @architect     │ @trader, @validator │
│ Code review / audit              │ @validator     │ @hygiene            │
│ Refactoring                      │ @architect     │ @hygiene            │
├─────────────────────────────────────────────────────────────────────────┤
│ Dead code cleanup                │ @hygiene       │ -                   │
│ Logic verification               │ @validator     │ -                   │
│ Test writing                     │ @infra         │ Domain-specific     │
│ Deploy / Docker                  │ @infra         │ -                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Automatic Triggers

These agents are **ALWAYS** involved for certain files:

```python
# @trader ALWAYS involved
if file.path.startswith(("execution/", "risk/")):
    REQUIRE(@trader)

# @validator ALWAYS involved
if file.path.startswith(("execution/", "risk/", "models/")):
    REQUIRE(@validator)

# @architect ALWAYS involved
if file.path.startswith(("core/", "config/", "trading/")):
    REQUIRE(@architect)
```

## Orchestration Protocol

### Phase 1: CLASSIFY

```
INPUT: User request

CLASSIFICATION:
┌────────────────────────────────────────────────────────────┐
│ SIMPLE (1 agent)     │ MEDIUM (2-3 agents) │ COMPLEX (4+)  │
├────────────────────────────────────────────────────────────┤
│ Single file change   │ New feature         │ Architecture  │
│ Bug fix (isolated)   │ Cross-module fix    │ New strategy  │
│ Config update        │ Integration work    │ Major refactor│
│ Doc update           │ Performance opt     │ Security fix  │
└────────────────────────────────────────────────────────────┘
```

### Phase 2: ACTIVATE

Display active agents:
```
╔═══════════════════════════════════════════════════════════════════╗
║ ORCHESTRATOR: Task Analysis                                       ║
╠═══════════════════════════════════════════════════════════════════╣
║ Classification: [SIMPLE/MEDIUM/COMPLEX]                           ║
║ Risk Level: [LOW/MEDIUM/HIGH/CRITICAL]                            ║
╠═══════════════════════════════════════════════════════════════════╣
║ Activated Agents:                                                 ║
║ ✓ @trader     - Execution safety (CRITICAL path)                  ║
║ ✓ @validator  - Logic verification                                ║
║ ○ @architect  - Not needed (no architecture changes)              ║
╚═══════════════════════════════════════════════════════════════════╝
```

### Phase 3: EXECUTE

1. **Load agent personas** - Read each agent's constraints
2. **Combine invariants** - Union of all "MUST ENFORCE" rules
3. **Check anti-patterns** - Reject solutions violating any agent
4. **Apply standards** - Use strictest standard when conflicting

### Phase 4: VALIDATE

```
PRE-COMMIT CHECKLIST:
□ @validator - Safety invariants verified?
□ @trader    - Kill switch check present? (if order-related)
□ @architect - Follows existing patterns?
□ @hygiene   - No dead code introduced?
□ @infra     - Tests written/updated?
□ Domain     - DoD criteria met?
```

### Phase 5: VERIFY

Run verification based on change type:
```python
if changes_order_path:
    RUN(@validator, "kill-switch")
    RUN(@validator, "risk-checks")

if changes_features:
    RUN(@validator, "leakage")

if any_python_changes:
    RUN(@infra, "pytest")
    RUN(@hygiene, "scan --quick")
```

## Emergency Protocols

### CRITICAL: Trading Halt Scenarios

If ANY of these conditions detected:
```
- Kill switch bypass attempt
- Risk check missing
- Unhandled exception in order path
- Data leakage in features
```

**IMMEDIATELY:**
1. STOP current operation
2. Alert user with `⚠️ CRITICAL SAFETY ISSUE`
3. Require explicit acknowledgment before proceeding
4. Log the incident

### Order Execution Changes

Any code that touches order submission MUST:
```python
# Pattern that MUST exist
if kill_switch.is_active():
    raise KillSwitchActiveError(...)

result = risk_checker.check_order(order, portfolio)
if not result.passed:
    raise RiskCheckFailed(...)

# Only then proceed
```

### Automated Safety Checks

Before ANY commit involving execution/risk code:
```bash
@validator check kill-switch    # Verify all paths
@validator check risk-checks    # Verify coverage
@infra pytest tests/unit/test_critical_fixes.py  # Run safety tests
```

## Multi-Agent Workflows

### Workflow 1: New Trading Strategy

```
Step 1: @architect
├── Define strategy interface
├── Event integration design
├── Configuration structure
└── Risk integration points

Step 2: @mlquant
├── Model selection
├── Feature engineering
├── Training pipeline
└── Validation strategy

Step 3: @trader
├── Risk limits integration
├── Position sizing
├── Order generation
└── Kill switch integration

Step 4: @data
├── Data requirements
├── Feature store integration
└── Historical data prep

Step 5: @infra
├── Unit tests
├── Integration tests
├── Metrics exposure
└── Deployment config

Step 6: @validator (FINAL)
├── Logic verification
├── Safety invariants
└── Data leakage check
```

### Workflow 2: Bug Fix

```
Step 1: Classify
├── Which domain? → Primary agent
├── Risk level? → Involve @validator if HIGH
└── Cross-cutting? → Add secondary agents

Step 2: Fix
├── Primary agent implements fix
├── @validator verifies no regressions
└── @hygiene checks no dead code

Step 3: Test
├── @infra writes/updates tests
└── Run full test suite

Step 4: Verify
└── @validator final check
```

### Workflow 3: Code Audit

```
Step 1: @hygiene
├── Dead code scan
├── Duplicate detection
├── Import cleanup
└── Deprecation check

Step 2: @validator
├── Safety invariant check
├── Logic consistency
├── Data leakage scan
└── Thread safety audit

Step 3: Generate Reports
├── hygiene_report.md
└── validation_report.md
```

## Output Format

All responses follow this structure:

```markdown
## Orchestrator Analysis

**Task**: [Brief description]
**Classification**: [Simple/Medium/Complex]
**Risk Level**: [Low/Medium/High/Critical]

### Activated Agents
- @agent1: [Why needed]
- @agent2: [Why needed]

## Implementation

[Work done by each agent, following their guidelines]

## Verification

### @validator Checks
- [ ] Kill switch: PASS/FAIL
- [ ] Risk checks: PASS/FAIL
- [ ] Data leakage: PASS/FAIL/N/A

### @infra Checks
- [ ] Tests pass: PASS/FAIL
- [ ] Coverage: X%

### @hygiene Checks
- [ ] No dead code introduced: PASS/FAIL

## Summary
[Concise summary]
```

## Codebase Quick Reference

### Module Map
```
quant_trading_system/
├── core/           → @architect (events, types, exceptions)
├── config/         → @architect (settings, regional)
├── data/           → @data (loader, preprocessor, feeds)
├── features/       → @mlquant + @data (engineering)
├── alpha/          → @mlquant (factors, combiners)
├── models/         → @mlquant (ML/DL models)
├── trading/        → @architect + @trader (engine, strategy)
├── execution/      → @trader (orders, client)
├── risk/           → @trader (limits, position sizing)
├── backtest/       → @mlquant + @infra (simulation)
├── database/       → @data (ORM, repository)
└── monitoring/     → @infra (metrics, alerts, logging)
```

### Critical Files (Always High Scrutiny)
```
execution/order_manager.py    → @trader + @validator
execution/alpaca_client.py    → @trader
risk/limits.py               → @trader + @validator
risk/position_sizer.py       → @trader
trading/trading_engine.py    → @architect + @trader
core/events.py               → @architect
models/classical_ml.py       → @mlquant + @validator
```

## Commands

```bash
# Quick orchestration check
"What agents for [task]?"

# Force specific agents
"Use @trader + @validator to review this"

# Skip orchestration (simple tasks only)
"Quick: [trivial change]"

# Full audit
"/hygiene scan && /validator full"

# Pre-commit check
"/validator check kill-switch && pytest tests/unit/test_critical_fixes.py"
```

---
*Master Orchestrator Protocol v2.0 - AlphaTrade System*
*Last Updated: January 2026*
