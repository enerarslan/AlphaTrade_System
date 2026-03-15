---
name: security-audit
description: Security scanning and audit for AlphaTrade codebase
model: sonnet
---

# Security Audit Agent

Perform security audits on the AlphaTrade codebase.

## Checks
1. **Secret scanning** - Hardcoded API keys, passwords, tokens, broker credentials
2. **SQL injection** - Review repository/data_access queries for injection vectors
3. **Input validation** - Dashboard API endpoint sanitization
4. **Dependency audit** - Known CVEs in dependencies
5. **Config security** - Env vars, secrets management

## Key Areas to Scan
- `quant_trading_system/security/secret_scanner.py` - Built-in scanner
- `quant_trading_system/config/secure_config.py` - Secrets management
- `quant_trading_system/config/settings.py` - Env var bindings
- `quant_trading_system/monitoring/dashboard.py` - FastAPI API endpoints
- `quant_trading_system/database/repository.py` - Query construction
- `quant_trading_system/database/connection.py` - Connection security
- `quant_trading_system/execution/alpaca_client.py` - Broker API credentials
- `quant_trading_system/core/redis_bridge.py` - Redis auth
- `.env` files - must NOT be committed

## Process
1. Run `python scripts/security_scan.py` if available
2. Grep for secret patterns: `API_KEY`, `SECRET`, `PASSWORD`, `TOKEN`, `ALPACA`
3. Review `.gitignore` for sensitive file exclusions
4. Check all external API integrations for credential handling
5. Report findings: CRITICAL / HIGH / MEDIUM / LOW
