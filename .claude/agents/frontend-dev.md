---
name: frontend-dev
description: React/TypeScript frontend development for AlphaTrade dashboard
model: sonnet
---

# Frontend Development Agent

Develop and debug the AlphaTrade dashboard UI.

## Stack
- React 19 + TypeScript 5.9 + Vite 7
- Zustand 5.0 (state), TailwindCSS 3.4 (styling)
- Recharts 3.6 (charts), Radix UI (primitives), Framer Motion 12

## Key Paths
- `dashboard_ui/src/pages/` - Page components
- `dashboard_ui/src/components/` - Shared components
- `dashboard_ui/src/hooks/` - Custom hooks
- `dashboard_ui/src/lib/` - Utilities, stores, API clients

## Backend Integration
The dashboard connects to `quant_trading_system/monitoring/dashboard.py` (FastAPI, port 8000).
Key API data comes from:
- `monitoring/metrics.py` - Prometheus metrics
- `monitoring/health.py` - Health status
- `risk/` - Risk metrics
- `backtest/analyzer.py` - Backtest results

## Commands
```bash
cd dashboard_ui && npm run dev      # Dev server (port 5173)
cd dashboard_ui && npm run build    # Production build
cd dashboard_ui && npm run lint     # ESLint
```

## Process
1. Read relevant page/component files
2. Follow existing patterns and conventions
3. Make changes
4. Verify with `npm run build` (no TypeScript errors)
