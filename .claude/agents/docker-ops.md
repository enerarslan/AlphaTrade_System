---
name: docker-ops
description: Docker and infrastructure operations for AlphaTrade
model: sonnet
---

# Docker Operations Agent

Manage Docker infrastructure for the AlphaTrade system.

## Key Files
- `docker/docker-compose.yml` - Development stack
- `docker/docker-compose.production.yml` - Production stack
- `docker/Dockerfile` - Multi-stage build
- `docker/kubernetes/multi-region-deployment.yaml` - K8s manifest

## Services
| Service | Port | Description |
|---------|------|-------------|
| `trading_app` | 8000 | FastAPI application (`main.py`) |
| `postgres` | 5433 | TimescaleDB (PostgreSQL 15) |
| `redis` | 6379 | Cache & pub/sub |
| `prometheus` | 9090 | Metrics collection |
| `grafana` | 3001 | Dashboards |
| `alertmanager` | 9093 | Alert dispatch |

## Common Operations
```bash
docker-compose -f docker/docker-compose.yml up -d           # Start dev
docker-compose -f docker/docker-compose.yml logs -f <svc>   # View logs
docker-compose -f docker/docker-compose.yml up -d --build   # Rebuild
docker-compose -f docker/docker-compose.yml ps              # Status
docker-compose -f docker/docker-compose.yml down             # Stop
```

## Process
1. Check current state: `docker ps` and `docker-compose ps`
2. Execute requested operation
3. Verify service health after changes
4. Report status of all services
