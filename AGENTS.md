# Repository Guidelines

Our goal is to build institutional level algorithmic trading program for myself.

## Project Structure & Module Organization
`quant_trading_system/` contains the backend domains (`alpha/`, `features/`, `models/`, `risk/`, `execution/`, `monitoring/`, `database/`, `config/`). This is the most important part of the project.
`main.py` is the primary CLI entrypoint for trading, backtests, training, data, and dashboard commands.  
`tests/unit/` and `tests/integration/` hold automated tests; shared fixtures live in `tests/conftest.py`.  
`dashboard_ui/` is the React + TypeScript + Vite frontend (`src/components`, `src/pages`, `src/lib`).  
Infra and ops assets are in `docker/`, `docs/`, `scripts/`, plus runtime folders such as `data/`, `models/`, and `logs/`.
There are 5 year stock data in data folder from 46 different stocks 

## Build, Test, and Development Commands
- `pip install -e ".[dev]"`: install backend package with lint/test tooling.
- `python main.py --help`: list available system commands.
- `python main.py dashboard --port 8000`: run backend dashboard API.
- `python main.py trade --mode paper`: start paper trading flow.
- `pytest` or `pytest -m "unit"`: run all tests or unit-only tests.
- `pytest -m "integration"`: run integration suite.
- `pytest --cov=quant_trading_system --cov-report=term-missing`: coverage report.
- `docker compose -f docker/docker-compose.yml up -d redis`: start Redis dependency.
- `cd dashboard_ui; npm ci; npm run dev`: install and run frontend locally.
- `cd dashboard_ui; npm run build` / `npm run lint`: frontend build and lint checks.

## Coding Style & Naming Conventions
Python follows Black/Ruff/isort with 100-char lines and 4-space indentation; mypy is configured in strict mode.  
Use `snake_case` for Python modules/functions, `PascalCase` for classes, and keep domain code in its matching package.  
Frontend code follows ESLint rules; use `PascalCase` for React components and keep page-route files consistent with existing lowercase/kebab patterns (for example `system-status.tsx`).

## Testing Guidelines
Use pytest markers: `unit`, `integration`, and `slow`.  
Name tests `test_*.py` and functions `test_*`; keep fast logic tests in `tests/unit/` and cross-service flows in `tests/integration/`.  
No coverage fail threshold is enforced in config, so contributors should maintain or improve coverage for touched modules and add regression tests with each bug fix.

## Commit & Pull Request Guidelines
Recent history contains many generic messages (`Update`, `Upd`); prefer explicit, imperative subjects with scope, e.g. `risk: tighten position limit checks`.  
PRs should include: concise summary, linked issue/ticket, risk notes, and exact validation commands run.  
For UI changes, attach screenshots; for schema updates, include Alembic migration files in `quant_trading_system/database/migrations/versions/`.

## Security & Configuration Tips
Keep secrets in `.env` and never commit credentials, logs, or generated artifacts.  
Run `python main.py health check --full` before merging operational changes.
