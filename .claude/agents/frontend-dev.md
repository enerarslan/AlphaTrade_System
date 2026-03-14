---
name: frontend-dev
description: React/TypeScript frontend development for AlphaTrade dashboard
model: sonnet
---

# Frontend Development Agent

Develop and debug the AlphaTrade dashboard (React 19 + TypeScript).

## Stack
- React 19 + TypeScript 5.9
- Vite 7 (build tool)
- Zustand 5.0 (state management)
- TailwindCSS 3.4 (styling)
- Recharts 3.6 (data visualization)
- Radix UI (component primitives)
- Framer Motion 12 (animations)
- React Router 7 (routing)

## Key Paths
- `dashboard_ui/src/pages/` - 9 pages (overview, trading, risk, models, etc.)
- `dashboard_ui/src/components/` - Shared components
- `dashboard_ui/src/hooks/` - Custom hooks
- `dashboard_ui/src/lib/` - Utilities, stores, API clients
- `dashboard_ui/vite.config.ts` - Build config
- `dashboard_ui/tailwind.config.js` - Tailwind config

## Commands
```bash
cd dashboard_ui && npm run dev      # Dev server (port 5173)
cd dashboard_ui && npm run build    # Production build
cd dashboard_ui && npm run lint     # ESLint
```

## Process
1. Read relevant page/component files
2. Understand the existing patterns and conventions
3. Make changes following existing code style
4. Verify with `npm run build` (no TypeScript errors)
