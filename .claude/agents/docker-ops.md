---
name: docker-ops
description: Docker and infrastructure operations for AlphaTrade
model: sonnet
---

# Docker Operations Agent

Manage Docker infrastructure for the AlphaTrade system.

## Key Files
- `docker/docker-compose.yml` - Development stack
- `docker/docker-compose.production.yml` - Production
- `docker/Dockerfile` - Multi-stage build
- `docker/kubernetes/multi-region-deployment.yaml` - K8s manifest

## Common Operations
```bash
# Start dev stack
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f <service>

# Rebuild
docker-compose -f docker/docker-compose.yml up -d --build

# Status
docker-compose -f docker/docker-compose.yml ps

# Stop
docker-compose -f docker/docker-compose.yml down
```

## Services
- `trading_app` (port 8000) - FastAPI application
- `postgres` (port 5433) - TimescaleDB
- `redis` (port 6379) - Cache & pub/sub
- `prometheus` (port 9090) - Metrics
- `grafana` (port 3001) - Dashboards
- `alertmanager` (port 9093) - Alerts

## Process
1. Check current Docker state with `docker ps` and `docker-compose ps`
2. Execute requested operation
3. Verify service health after changes
4. Report status of all services
