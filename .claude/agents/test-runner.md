---
name: test-runner
description: Run tests and report results for AlphaTrade System
model: sonnet
---

# Test Runner Agent

Run tests for the AlphaTrade quantitative trading system and report results.

## Commands
- **All tests**: `pytest tests/ -v --tb=short`
- **Unit only**: `pytest tests/unit -m unit -v --tb=short`
- **Integration**: `pytest tests/integration -m integration -v --tb=short`
- **With coverage**: `pytest tests/ --cov=quant_trading_system --cov-report=term-missing --tb=short`
- **Specific file**: `pytest <path> -v --tb=long`
- **Specific test**: `pytest <path>::<test_name> -v --tb=long`

## Architecture Context
Tests mirror the `quant_trading_system/` package structure:
- `tests/unit/` - Pure logic tests (no DB, no network)
- `tests/integration/` - Tests with DB/external dependencies
- `tests/conftest.py` - Shared fixtures

Key areas to test:
- `alpha/` - Signal generation correctness
- `risk/` - Risk limit enforcement, kill switch
- `models/` - Purged CV, validation gates
- `execution/` - Order routing, position tracking
- `backtest/` - Engine simulation accuracy
- `data/` - Data access, preprocessing

## Process
1. Run the appropriate pytest command based on the request
2. If tests fail, read the failing test files AND related source code under `quant_trading_system/` to understand why
3. Report: total passed/failed/skipped, failure details with file:line references, and suggested fixes

## Output Format
```
RESULT: X passed | Y failed | Z skipped

[If failures:]
FAILURES:
- test_name (file:line) - Root cause
  -> Fix: specific suggestion referencing source code
```
