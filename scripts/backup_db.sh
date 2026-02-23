#!/usr/bin/env bash
set -euo pipefail

# Daily SQLite snapshot backup with retention.
ROOT_DIR="/Users/nikolas/Documents/ТЗ : Ascend Edge Ltd/TZ---AEL"
DB_PATH="$ROOT_DIR/crm.db"
BACKUP_DIR="$ROOT_DIR/backups"
RETENTION_DAYS="${RETENTION_DAYS:-14}"

mkdir -p "$BACKUP_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_FILE="$BACKUP_DIR/crm_${TS}.sqlite3"

sqlite3 "$DB_PATH" ".backup '$OUT_FILE'"
gzip -f "$OUT_FILE"

# Cleanup old backups
find "$BACKUP_DIR" -type f -name "crm_*.sqlite3.gz" -mtime +"$RETENTION_DAYS" -delete

echo "Backup created: ${OUT_FILE}.gz"
