---
active: true
iteration: 1
max_iterations: 100
completion_promise: "ALL_PHASES_COMPLETE"
started_at: "2025-12-29T23:42:10Z"
---


## MISSION: JPMorgan-Level Algorithmic Trading System - Full Validation & Autonomous Development

You are an expert quantitative developer tasked with validating, testing, and completing a JPMorgan-level algorithmic trading system. Work autonomously until ALL objectives are complete.


### ARCHITECTURE REFERENCE
Read and strictly follow: ARCHITECTURE.md

### PHASE 1: ENVIRONMENT SETUP & VALIDATION
1. Navigate to project directory
2. Activate virtual environment: .\venv\Scripts\activate
3. Verify Python 3.11+ is installed
4. Install all dependencies: pip install -r requirements.txt --upgrade
5. Install additional dependencies if missing (psutil, ta-lib if available)
6. Verify Redis is running (if not, start it from redis\ folder or Docker)
7. Verify PostgreSQL + TimescaleDB is available (use Docker if needed)
8. Create .env file with required environment variables if missing

### PHASE 2: DATABASE SETUP
1. Run database migrations: alembic upgrade head
2. Verify all tables are created
3. Check TimescaleDB hypertables are configured
4. Test database connection from Python code
5. If database errors, fix connection strings and retry

### PHASE 3: RUN ALL UNIT TESTS
1. Run: pytest tests/unit -v --cov=quant_trading_system --cov-report=html
2. Analyze test failures carefully
3. Fix ALL failing tests - DO NOT skip or ignore any
4. Re-run tests until ALL pass with >80% coverage
5. Document any issues found and fixes applied

### PHASE 4: RUN INTEGRATION TESTS
1. Run: pytest tests/integration -v
2. Fix any integration test failures
3. Verify data pipeline works end-to-end
4. Test feature computation pipeline
5. Test model prediction pipeline

### PHASE 5: DATA VALIDATION
1. Load and validate all 46 CSV files from data\raw2. Check for:
   - Missing values
   - Data gaps
   - Invalid prices (negative, zeros)
   - Timestamp ordering
   - OHLC relationships (low <= open,close <= high)
3. Fix any data issues programmatically
4. Generate data quality report

### PHASE 6: FEATURE ENGINEERING VALIDATION
1. Test all technical indicators (RSI, MACD, Bollinger Bands, etc.)
2. Test statistical features
3. Test cross-sectional features
4. Verify no look-ahead bias
5. Test feature store caching (Redis)
6. Benchmark feature computation time

### PHASE 7: MODEL TRAINING & VALIDATION
1. Train XGBoost model with proper train/validation split
2. Train LightGBM model
3. Train LSTM model (use GPU if available via torch.cuda)
4. Implement walk-forward validation
5. Calculate and log:
   - Sharpe ratio
   - Information coefficient
   - AUC/Accuracy
6. Save trained models to models\ directory
7. If model performance < targets, tune hyperparameters

### PHASE 8: BACKTESTING
1. Run full backtest: python scripts/run_backtest.py --start 2020-01-01 --end 2024-12-31
2. Verify realistic slippage and transaction costs
3. Calculate all performance metrics:
   - Total return
   - Sharpe ratio (target > 1.5)
   - Max drawdown (target < 15%)
   - Win rate (target > 52%)
   - Profit factor (target > 1.3)
4. Generate equity curve visualization
5. Generate performance report
6. If metrics below targets, investigate and improve strategy

### PHASE 9: RISK MANAGEMENT VALIDATION
1. Test position sizing algorithms
2. Test portfolio optimization
3. Test risk limits enforcement
4. Test kill switch functionality
5. Verify VaR calculations

### PHASE 10: PAPER TRADING SIMULATION
1. Start paper trading: python main.py trade --mode paper
2. Verify connection to Alpaca (or mock if credentials unavailable)
3. Test order submission
4. Test position tracking
5. Test real-time data feed
6. Run for at least 100 simulated bars

### PHASE 11: MONITORING & DASHBOARD
1. Start dashboard: python main.py dashboard --port 8000
2. Verify Prometheus metrics endpoint
3. Verify all API endpoints respond
4. Test alerting system

### PHASE 12: FINAL VALIDATION
1. Run full test suite one more time
2. Generate comprehensive test report
3. Document all fixes and improvements made
4. Create VALIDATION_REPORT.md with:
   - All test results
   - Performance metrics
   - Issues found and resolved
   - Recommendations for production

### CRITICAL RULES
- NEVER skip a failing test - fix it
- NEVER use mock data for backtesting - use real data from data\raw\ or download it from alpaca API
- ALWAYS use proper train/test splits (no data leakage)
- ALWAYS save logs to logs\ directory
- USE GPU for deep learning if torch.cuda.is_available(), it must be available, if not, fix it
- USE Redis for caching if available
- Document everything in markdown files

### COMPLETION CRITERIA
Output <promise>ALL_PHASES_COMPLETE</promise> ONLY when:
- All unit tests pass
- All integration tests pass
- Backtesting achieves Sharpe > 1.0
- Paper trading simulation runs without errors
- Dashboard starts successfully
- VALIDATION_REPORT.md is created

### IF STUCK AFTER 30 ITERATIONS
- Document all blocking issues in BLOCKING_ISSUES.md
- List all attempted solutions
- Provide specific error messages
- Suggest manual intervention steps needed

### PROGRESS TRACKING
After each phase, create/update PROGRESS.md with:
- [x] Completed phases
- [ ] Pending phases
- Current status
- Issues encountered
- Estimated remaining work

