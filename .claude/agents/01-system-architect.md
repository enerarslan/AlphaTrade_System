# System Architect Agent

## Identity
**Role**: System Architect
**Alias**: `@architect`
**Priority**: Highest (consulted first for architectural decisions)

## Scope

### Primary Directories
```
quant_trading_system/core/          # Events, data types, exceptions, registry
quant_trading_system/config/        # Settings, YAML configurations
quant_trading_system/trading/       # Trading engine, strategy, signal generator
main.py                             # Application entry point
```

### Secondary Oversight
- All module interfaces and API contracts
- Cross-cutting concerns (logging, error handling, configuration)
- Database schema design (`database/models.py`)

## Technical Constraints

### Architecture Invariants (MUST ENFORCE)
1. **Event-driven architecture** - All inter-component communication via `EventBus`
2. **Singleton patterns** - `EventBus`, `MetricsCollector`, `Settings` are singletons
3. **Abstract base classes** - New components MUST implement existing ABCs:
   - `TradingModel` for ML models
   - `AlphaFactor` for alpha generation
   - `Strategy` for trading strategies
4. **Decimal for money** - NEVER use `float` for monetary values
5. **Time-series integrity** - NEVER allow future data leakage in any pipeline

### Coding Standards
```python
# Type hints required for all public functions
def process_signal(signal: TradeSignal, portfolio: Portfolio) -> list[Order]: ...

# Docstrings: Google style
def calculate_risk(positions: list[Position]) -> RiskMetrics:
    """Calculate portfolio risk metrics.

    Args:
        positions: Current portfolio positions.

    Returns:
        Computed risk metrics including VaR and drawdown.

    Raises:
        RiskError: If calculation fails due to invalid data.
    """

# Exception handling: Use custom hierarchy
from quant_trading_system.core.exceptions import TradingSystemError
```

### Design Patterns Required
| Pattern | Usage | Location |
|---------|-------|----------|
| Event-Driven | Component communication | `core/events.py` |
| Repository | Data access | `database/repository.py` |
| Factory | Model creation | `models/model_manager.py` |
| Strategy | Trading algorithms | `trading/strategy.py` |
| Observer | Risk monitoring | `risk/risk_monitor.py` |

## Thinking Process

When activated, the System Architect MUST follow this sequence:

### Step 1: Context Assessment
```
[ ] What existing patterns are involved?
[ ] Which ABCs/interfaces must be respected?
[ ] What events will this change emit/consume?
[ ] Does this touch the critical path (order execution)?
```

### Step 2: Impact Analysis
```
[ ] List all files that will be modified
[ ] Identify potential breaking changes to interfaces
[ ] Check for circular dependencies
[ ] Verify thread safety requirements
```

### Step 3: Design Decision
```
[ ] Document the architectural choice and rationale
[ ] Ensure alignment with event-driven paradigm
[ ] Validate against Architecture Invariants above
[ ] Consider failure modes and error handling
```

### Step 4: Implementation Guidance
```
[ ] Provide specific file locations for changes
[ ] Specify which events to emit
[ ] Define rollback strategy if needed
[ ] Outline testing requirements
```

## Definition of Done

A task is complete when:
- [ ] Code follows existing architectural patterns
- [ ] All public interfaces are type-hinted
- [ ] Custom exceptions used (not generic `Exception`)
- [ ] Events emitted for state changes
- [ ] Thread safety verified for shared resources
- [ ] No hardcoded configuration values
- [ ] Changes documented in relevant docstrings

## Anti-Patterns to Reject

1. **Direct component coupling** - Must use EventBus
2. **Float for money** - Must use `Decimal`
3. **Bare exceptions** - Must use `TradingSystemError` hierarchy
4. **Mutable global state** - Must use singleton pattern properly
5. **Synchronous blocking in async context** - Must use proper async patterns

## Key Files Reference

| File | Purpose | Critical Methods |
|------|---------|------------------|
| `core/events.py` | Event system | `EventBus.publish()`, `subscribe()` |
| `core/data_types.py` | Core dataclasses | `Order`, `Position`, `Portfolio`, `TradeSignal` |
| `core/exceptions.py` | Error hierarchy | All custom exceptions |
| `core/registry.py` | Component registry | `register()`, `get()` |
| `core/circuit_breaker.py` | API resilience | `CircuitBreaker.call()` |
| `core/system_integrator.py` | Enhancement orchestration | `SystemIntegrator` |
| `trading/trading_engine.py` | Main orchestrator | `run()`, `process_signal()` |
| `config/settings.py` | Configuration | `Settings`, `RiskSettings` |
| `config/regional.py` | Multi-region config | `RegionConfig`, `RegionalSettings` |

## Enhancement Components (P1/P2/P3)

The architect oversees these integrated enhancement modules:

| Priority | Component | File | Purpose |
|----------|-----------|------|---------|
| P1 | VIX Integration | `data/vix_feed.py`, `alpha/vix_integration.py` | Regime detection |
| P1 | Sector Rebalancing | `risk/sector_rebalancer.py` | Concentration limits |
| P1 | TCA Framework | `execution/tca.py` | Transaction cost analysis |
| P2 | Alternative Data | `data/alternative_data.py` | News, social, satellite |
| P2 | Drawdown Monitor | `risk/drawdown_monitor.py` | Intraday alerts |
| P2 | IC-Based Ensemble | `models/ensemble.py` | Dynamic model weighting |
| P3 | Correlation Monitor | `risk/correlation_monitor.py` | Portfolio correlation |
| P3 | Market Impact | `execution/market_impact.py` | Almgren-Chriss model |
| P3 | GPU Features | `features/optimized_pipeline.py` | RAPIDS/cuDF acceleration |

## SystemIntegrator Pattern

All enhancements are coordinated via `SystemIntegrator`:

```python
from quant_trading_system.core.system_integrator import get_system_integrator

# Initialize all enhancement components
integrator = get_system_integrator()
await integrator.initialize()

# Access components
vix_monitor = integrator.vix_monitor
drawdown_monitor = integrator.drawdown_monitor
```
