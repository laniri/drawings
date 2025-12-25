# POST /api/v1/backup/database

## Summary
Create database backup

## Description
Create a database-only backup with enhanced data integrity verification. The backup service automatically detects and handles different SQLite database URL formats:

- `sqlite:///absolute/path/to/database.db` (recommended format)
- `sqlite://relative/path/to/database.db` (alternative format)
- `sqlite://:memory:` (in-memory databases - limited backup support)

**Enhanced Backup Process:**
- **Transaction Integrity**: Ensures all pending database transactions are committed before backup
- **WAL Checkpoint**: Flushes Write-Ahead Log (WAL) to main database file for complete data capture
- **Data Verification**: Validates backup contains actual data, not just valid SQLite structure
- **Retry Logic**: Implements retry mechanism for handling database locks during backup
- **Content Validation**: Verifies table existence and row counts to ensure backup completeness

For in-memory databases, the backup service will log appropriate warnings as direct file backup is not possible.

## Parameters
No parameters

## Responses
- **200**: Successful Response
  ```json
  {
    "backup_name": "db_backup_20241225_143022.db",
    "backup_path": "backups/db_backup_20241225_143022.db",
    "timestamp": "20241225_143022",
    "size_bytes": 2048576,
    "size_mb": 1.95,
    "type": "database_only",
    "status": "completed"
  }
  ```
- **500**: Backup failed (e.g., in-memory database, insufficient permissions)

## Example
```http
POST /api/v1/backup/database
```

## Notes
- **Enhanced Data Integrity**: Backup process now includes transaction commit verification and WAL checkpoint flushing
- **Content Validation**: Backups are verified to contain actual data, not just valid SQLite structure
- **Retry Mechanism**: Automatic retry logic handles database locks during backup operations
- **Backup operations may be limited for in-memory databases (`:memory:`)**
- **The service automatically handles different SQLite URL formats**
- **Check logs for warnings if using non-standard database configurations**
- **Backup verification includes table count and row count validation**
