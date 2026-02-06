# JPMorgan-Grade Dashboard Production Stack

This stack adds a hardened edge layer for the dashboard:

- Reverse proxy (NGINX) with TLS termination
- API route isolation (`/api/*`) and WebSocket proxying (`/api/ws/*`)
- Edge rate limiting and connection limiting
- SSO/OIDC federation support
- Centralized SIEM forwarding for signed audit records

## Compose Profiles

Base stack:

- `docker/docker-compose.yml`

Production override:

- `docker/docker-compose.production.yml`

Start production profile:

```bash
python main.py deploy docker up --profile production
```

Check status:

```bash
python main.py deploy docker status --profile production
```

Stop:

```bash
python main.py deploy docker down --profile production
```

## Edge Services

- `reverse_proxy`: TLS + routing + edge controls
- `dashboard_ui`: React static build served by NGINX
- `dashboard_api`: FastAPI dashboard process
- `certbot` (optional, profile `tls`): automatic renewal loop

## TLS Behavior

- If valid certificates are present under `/etc/nginx/certs/live/<domain>/`, they are used.
- Otherwise, reverse proxy auto-generates a short-lived self-signed cert at startup.

For first-time Let's Encrypt issuance (example):

```bash
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.production.yml run --rm certbot \
  certonly --webroot -w /var/www/certbot \
  --config-dir /etc/nginx/certs --work-dir /var/www/certbot/work --logs-dir /var/www/certbot/logs \
  -d your-dashboard-domain.example --email you@example.com --agree-tos --no-eff-email
```

Then restart reverse proxy:

```bash
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.production.yml restart reverse_proxy
```

## Required Security Environment

Set these before production launch:

- `JWT_SECRET_KEY`
- `DASHBOARD_USERS`
- `DASHBOARD_USER_ROLES`
- `DASHBOARD_AUDIT_SECRET`
- `REQUIRE_AUTH=true`

Recommended:

- `OIDC_ENABLED=true`
- `OIDC_ISSUER_URL`, `OIDC_CLIENT_ID`, `OIDC_CLIENT_SECRET`, `OIDC_REDIRECT_URI`
- `OIDC_ROLE_MAP` (example: `traders:operator,riskdesk:risk,platform-admin:admin`)
- `DASHBOARD_SIEM_ENABLED=true`
- `DASHBOARD_SIEM_ENDPOINT`, `DASHBOARD_SIEM_API_KEY`

## API Security Controls Enabled

- App-level rate limiting (`DASHBOARD_RATE_LIMIT_*`)
- Edge-level NGINX rate limit / connection limit
- MFA for privileged control actions
- Signed audit trail + export + SIEM forwarding endpoints
- OIDC SSO login endpoints (`/auth/sso/*`)
