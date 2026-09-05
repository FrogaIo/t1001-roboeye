#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <mac-lan-ip>"
  echo "Example: $0 192.168.1.42"
  exit 1
fi

LAN_IP="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CERT_DIR="$PROJECT_DIR/certs"

mkdir -p "$CERT_DIR"

openssl genrsa -out "$CERT_DIR/roboeye-ca.key" 2048
openssl req -x509 -new -sha256 -days 3650 \
  -key "$CERT_DIR/roboeye-ca.key" \
  -out "$CERT_DIR/roboeye-ca.crt" \
  -subj "/CN=RoboEye Local CA" \
  -addext "basicConstraints=critical,CA:TRUE" \
  -addext "keyUsage=critical,keyCertSign,cRLSign"

openssl genrsa -out "$CERT_DIR/server.key" 2048
openssl req -new \
  -key "$CERT_DIR/server.key" \
  -out "$CERT_DIR/server.csr" \
  -subj "/CN=$LAN_IP"

EXT_FILE="$CERT_DIR/server.ext"
printf '%s\n' \
  "subjectAltName=IP:$LAN_IP" \
  "basicConstraints=critical,CA:FALSE" \
  "keyUsage=critical,digitalSignature,keyEncipherment" \
  "extendedKeyUsage=serverAuth" > "$EXT_FILE"

openssl x509 -req -sha256 -days 825 \
  -in "$CERT_DIR/server.csr" \
  -CA "$CERT_DIR/roboeye-ca.crt" \
  -CAkey "$CERT_DIR/roboeye-ca.key" \
  -CAcreateserial \
  -out "$CERT_DIR/server.crt" \
  -extfile "$EXT_FILE"

echo
echo "Certificates created in $CERT_DIR"
echo "Install roboeye-ca.crt on the phone and trust it."
echo "Then open: https://$LAN_IP:8443"
