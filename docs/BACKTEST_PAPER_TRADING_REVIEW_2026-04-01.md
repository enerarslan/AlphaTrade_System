# Backtest And Paper Trading Implementation Update

Date: 2026-04-01

Scope: implement the critical findings from the backtest and paper-trading audit so the repository uses the promoted model contract more consistently across `train -> promotion package -> backtest -> replay -> paper trading`.

## Summary

The highest-risk parity breaks identified in the audit are now closed in code:

- `main.py backtest` and `main.py trade` now accept `--promotion-package`
- `scripts/trade.py` was replaced with a current-API, artifact-aware wrapper built on `create_trading_engine(...)`
- promotion-package score conversion for ranker models now matches the train-time raw-score -> sigmoid policy
- promotion-package execution costs now propagate more completely into backtest and replay simulators
- paper trading no longer liquidates positions just because a fresh signal was absent on one bar
- live/paper now tracks position hold contracts and can exit on horizon / TP / SL / max-holding rules
- backtest annualization and holding-period accounting are now timeframe-aware and symbol-aware

This does not guarantee profitability. It materially improves correctness, parity, and risk realism, which is the necessary prerequisite before model ranking or paper-PnL numbers can be trusted.

## Implemented Fixes

### 1. Operator Surface Is Now Promotion-Package Aware

Implemented:

- `main.py trade` now exposes `--promotion-package`
- `main.py backtest` now exposes `--promotion-package`
- `scripts/trade.py` now uses the current package-layer engine factory instead of stale config wiring

Result:

- the main CLI can now run artifact-backed backtests and paper/live sessions from the promoted package contract
- the previous broken wrapper path that instantiated outdated `TradingEngineConfig` fields is removed

Key files:

- `main.py`
- `scripts/trade.py`

### 2. Paper Trading Uses The Promoted Model Contract More Faithfully

Implemented:

- `scripts/trade.py` now loads the promotion package directly
- package symbols can override the default symbol set when the operator did not explicitly choose a custom universe
- package timeframe is pushed into engine runtime config
- package sizing contract is pushed into portfolio sizing:
  - `max_position_pct`
  - `max_total_positions`
  - `confidence_position_sizing`
  - `min_confidence_position_scale`
- package-driven paper trading now registers an external signal source that computes features and signals from the promotion package

Result:

- paper trading now consumes the promoted artifact path instead of relying on stale strategy-only wrapper behavior

Key files:

- `scripts/trade.py`

### 3. Backtest / Replay Score Conversion Now Matches Training Better

Implemented:

- `quant_trading_system/backtest/promotion.py` now treats ranker-style models using the same raw-score clipping plus sigmoid transform used in training
- `predict_proba` models still use native probability output
- raw `predict(...)` fallback models now map deterministically into `[0, 1]` without per-batch standardization

Result:

- `lightgbm_ranker` thresholds are now materially more portable between training and replay/backtest
- the old symbol-local batch normalization path that distorted inference semantics is removed

Key files:

- `quant_trading_system/backtest/promotion.py`

### 4. Execution Cost Contract Propagates Into Simulation More Cleanly

Implemented:

- `scripts/backtest.py` now carries package `spread_bps`, `slippage_bps`, and `impact_bps`
- backtest realistic and pessimistic simulators now receive those values instead of only partial defaults
- replay now constructs an explicit market simulator using promotion-package spread/slippage inputs
- replay combines package slippage and impact into the effective slippage path when the scenario is still on defaults

Result:

- backtest and replay execution realism is closer to the promoted package cost contract used during model selection

Key files:

- `scripts/backtest.py`
- `quant_trading_system/backtest/replay.py`

### 5. Paper / Live Exit Semantics Are Closer To Backtest

Implemented:

- `PortfolioManager.check_rebalance_needed(...)` no longer treats an empty fresh target as an implicit liquidation order outside `SIGNAL_DRIVEN` mode
- order metadata now preserves hold-contract fields needed downstream:
  - `signal_horizon_bars`
  - `signal_direction`
  - `signal_timestamp`
  - existing signal metadata passthrough remains intact
- `TradingEngine` now tracks per-symbol hold contracts after fills and evaluates exits for:
  - `take_profit`
  - `stop_loss`
  - `horizon`
  - `max_holding`

Result:

- paper/live no longer churns out of positions just because there was no fresh signal on a single iteration
- exit behavior is now much closer to the backtest contract

Key files:

- `quant_trading_system/trading/portfolio_manager.py`
- `quant_trading_system/trading/trading_engine.py`

### 6. Trading Engine Now Respects Fresh-Bar Semantics Better

Implemented:

- market-hour signal generation now uses only symbols that actually received a fresh bar in the current iteration
- hold-contract bar counters advance only for symbols with a fresh bar
- backtest holding periods also advance only for symbols that actually updated

Result:

- sparse data / missing bars are less likely to create false horizon exits or distorted holding periods

Key files:

- `quant_trading_system/trading/trading_engine.py`
- `quant_trading_system/backtest/engine.py`

### 7. Intraday Backtest Annualization Is Timeframe-Aware

Implemented:

- `BacktestEngine` now estimates annualization periods from observed bar spacing
- intraday market-condition volatility no longer assumes a flat `sqrt(252)` annualization for all datasets

Result:

- volatility-aware execution realism is more correct for `15Min`, `5Min`, and other intraday paths

Key files:

- `quant_trading_system/backtest/engine.py`

## Added Regression Coverage

Direct tests were added for the newly implemented behavior:

- CLI parser coverage for `trade --promotion-package` and `backtest --promotion-package`
- ranker score transform parity in the promotion adapter
- backtest holding-period advancement only on updated symbols
- replay propagation of promotion-package execution costs into the market simulator
- portfolio-manager protection against empty-target forced liquidation
- order-request propagation of signal horizon metadata
- trading-engine horizon exit submission for tracked hold contracts
- artifact-aware `scripts/trade.py` configuration and promotion signal-source behavior

## Validation Run

Executed and passed:

```bash
python -m pytest tests/unit/test_main.py tests/unit/test_backtest.py tests/unit/test_backtest_script.py tests/unit/test_backtest_replay.py tests/unit/test_trading.py tests/unit/test_trade_script.py -q -x -s
```

Result:

- `164 passed`

## Remaining Risks

The code path is materially better, but these items are still operationally important:

- full historical reruns were not executed in this change set, so no new claim is made about absolute PnL superiority
- broker-specific paper/live frictions still depend on real Alpaca behavior, websocket timing, and fill quality outside unit tests
- promotion-package profitability still depends on model quality, labeling quality, universe construction, and realistic turnover
- no guarantee can be made that the system will "make money"; what is improved here is parity, correctness, and execution realism

## Recommended Next Step

Now that the contract is aligned, the next practical step is not another wrapper change. It is a full artifact-driven comparison run for the promoted candidates:

1. rerun backtest on the promotion packages for the current `lightgbm_ranker` and `tcn` candidates
2. run replay with the same promotion packages and inspect execution SLOs
3. paper trade only the candidates that remain strong after replay under the new parity rules

That will give a much cleaner answer to which model actually survives realistic deployment conditions.
