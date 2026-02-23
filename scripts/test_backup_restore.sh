#!/usr/bin/env bash
set -euo pipefail

# Verifies latest backup can be restored and queried.
ROOT_DIR="/Users/nikolas/Documents/ТЗ : Ascend Edge Ltd/TZ---AEL"
BACKUP_DIR="$ROOT_DIR/backups"
TMP_RESTORE="$ROOT_DIR/backups/_restore_test.sqlite3"

LATEST_BACKUP="$(ls -1t "$BACKUP_DIR"/crm_*.sqlite3.gz 2>/dev/null | head -n1 || true)"

if [[ -z "$LATEST_BACKUP" ]]; then
  echo "No backups found in $BACKUP_DIR"
  exit 1
fi

gzip -dc "$LATEST_BACKUP" > "$TMP_RESTORE"
sqlite3 "$TMP_RESTORE" "SELECT 1;" >/dev/null

echo "Restore test passed for: $LATEST_BACKUP"
rm -f "$TMP_RESTORE"
