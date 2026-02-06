#!/usr/bin/env sh
set -eu

CERT_DIR="/etc/nginx/certs"
DOMAIN="${DASHBOARD_DOMAIN:-localhost}"

mkdir -p "${CERT_DIR}"

LIVE_CERT="${CERT_DIR}/live/${DOMAIN}/fullchain.pem"
LIVE_KEY="${CERT_DIR}/live/${DOMAIN}/privkey.pem"
CERT_FILE="${CERT_DIR}/fullchain.pem"
KEY_FILE="${CERT_DIR}/privkey.pem"

if [ -s "${LIVE_CERT}" ] && [ -s "${LIVE_KEY}" ]; then
  cp "${LIVE_CERT}" "${CERT_FILE}"
  cp "${LIVE_KEY}" "${KEY_FILE}"
fi

if [ ! -s "${CERT_FILE}" ] || [ ! -s "${KEY_FILE}" ]; then
  echo "[edge] generating self-signed certificate for ${DOMAIN}"
  openssl req -x509 -nodes -newkey rsa:2048 -days 30 \
    -subj "/CN=${DOMAIN}" \
    -addext "subjectAltName=DNS:${DOMAIN},DNS:localhost,IP:127.0.0.1" \
    -keyout "${KEY_FILE}" \
    -out "${CERT_FILE}" >/dev/null 2>&1
fi

exec "$@"
