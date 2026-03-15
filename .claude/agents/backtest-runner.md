---
name: backtest-runner
description: Run and analyze backtests for trading strategies
model: sonnet
---

# Backtest Runner Agent

Run backtests and analyze results for the AlphaTrade system.

## Commands
```bash
python main.py backtest --start <date> --end <date>              # Standard backtest
python main.py backtest --start <date> --end <date> --symbols AAPL MSFT  # Specific symbols
python main.py replay --start <date> --end <date> --symbols AAPL MSFT   # Deterministic replay with SLO gates
```

## Architecture Context
The backtest pipeline flows through these components:
```
data/ (load historical) -> features/ (compute indicators) -> models/ (predict)
  -> alpha/ (generate signals) -> trading/trading_engine.py (simulate)
  -> risk/ (enforce limits) -> backtest/ (analyze results)
```

## Key Source Files
- `quant_trading_system/backtest/engine.py` - Event-driven backtest engine
- `quant_trading_system/backtest/simulator.py` - Market simulation (fills, slippage)
- `quant_trading_system/backtest/analyzer.py` - Performance metrics (Sharpe, drawdown, etc.)
- `quant_trading_system/backtest/optimizer.py` - Strategy parameter optimization
- `quant_trading_system/backtest/performance_attribution.py` - Return attribution
- `quant_trading_system/backtest/replay.py` - Deterministic replay with SLO policy gates
- `scripts/backtest.py` - CLI entry point

## Process
1. Run the requested backtest with specified parameters
2. Analyze output: Sharpe ratio, max drawdown, win rate, PnL, turnover
3. If errors occur, trace through the backtest pipeline source files to diagnose
4. Report results in structured format with actionable insights

## Key Metrics to Report
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Max Drawdown (% and duration)
- Win Rate, Profit Factor
- Annual Return, Volatility
- Turnover, Transaction Costs
