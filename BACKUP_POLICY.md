# Backup policy (SQLite)

## Scope
- Primary DB file: `crm.db`
- Backup target directory: `backups/`

## Schedule
- Frequency: daily snapshot
- Retention: 14 days (configurable via `RETENTION_DAYS`)

## Commands
- Create backup:
  - `bash /Users/nikolas/Documents/ТЗ : Ascend Edge Ltd/TZ---AEL/scripts/backup_db.sh`
- Validate restore:
  - `bash /Users/nikolas/Documents/ТЗ : Ascend Edge Ltd/TZ---AEL/scripts/test_backup_restore.sh`

## Suggested cron
```cron
0 2 * * * bash /Users/nikolas/Documents/ТЗ\ :\ Ascend\ Edge\ Ltd/TZ---AEL/scripts/backup_db.sh >> /Users/nikolas/Documents/ТЗ\ :\ Ascend\ Edge\ Ltd/TZ---AEL/bot_runtime.log 2>&1
30 2 * * * bash /Users/nikolas/Documents/ТЗ\ :\ Ascend\ Edge\ Ltd/TZ---AEL/scripts/test_backup_restore.sh >> /Users/nikolas/Documents/ТЗ\ :\ Ascend\ Edge\ Ltd/TZ---AEL/bot_runtime.log 2>&1
```

## Recovery
1. Pick the latest `crm_*.sqlite3.gz` from `backups/`.
2. Decompress to target file:
   - `gzip -dc backups/crm_YYYYMMDD_HHMMSS.sqlite3.gz > crm_restored.db`
3. Verify:
   - `sqlite3 crm_restored.db "SELECT 1;"`
