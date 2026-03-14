---
name: lint-check
description: Run linting, formatting, and type checking for AlphaTrade
model: haiku
---

# Lint & Type Check Agent

Run code quality checks for the AlphaTrade system.

## Process
1. Run `ruff check .` for linting
2. Run `black --check .` for format checking
3. Run `mypy quant_trading_system/` for type checking
4. Summarize all issues found with file:line references
5. If `--fix` is requested, run `ruff check . --fix` and `black .` to auto-fix

## Output
Report each tool's results concisely. Group issues by severity.
