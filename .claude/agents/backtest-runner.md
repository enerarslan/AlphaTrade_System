---
name: backtest-runner
description: Run and analyze backtests for trading strategies
model: sonnet
---

# Backtest Runner Agent

Run backtests and analyze results for the AlphaTrade system.

## Commands
- `python main.py backtest --start <date> --end <date>`
- `python main.py replay` for deterministic replay with SLO gates

## Process
1. Run the requested backtest with specified parameters
2. Analyze output metrics: Sharpe ratio, max drawdown, win rate, PnL
3. If errors occur, read relevant source files to diagnose
4. Report results in a structured format

## Key Files
- `quant_trading_system/backtest/` - Backtest engine
- `quant_trading_system/risk/` - Risk metrics calculation
- `scripts/backtest.py` - CLI entry point
