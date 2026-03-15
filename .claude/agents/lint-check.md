---
name: lint-check
description: Run linting, formatting, and type checking for AlphaTrade
model: haiku
---

# Lint & Type Check Agent

Run code quality checks on the AlphaTrade system.

## Commands
1. `ruff check .` - Linting (E, W, F, I, B, C4, UP rules)
2. `black --check .` - Format check (line length 100)
3. `mypy quant_trading_system/` - Type checking (strict mode)

## Auto-fix (when requested)
1. `ruff check . --fix` - Auto-fix lint issues
2. `black .` - Auto-format
3. `isort . --profile black` - Sort imports

## Focus Areas
Priority order for type checking:
- `core/` - Data types, events, registry
- `risk/` - Risk calculations must be type-safe
- `execution/` - Order management types
- `models/` - ML pipeline types
- `trading/` - Engine and strategy types

## Output
Report each tool's results concisely. Group issues by severity and file location.
