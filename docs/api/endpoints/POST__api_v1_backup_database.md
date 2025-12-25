# POST /api/v1/backup/database

## Summary
Create database backup

## Description
Create a database-only backup. The backup service automatically detects and handles different SQLite database URL formats:

- `sqlite:///absolute/path/to/database.db` (recommended format)
- `sqlite://relative/path/to/database.db` (alternative format)
- `sqlite://:memory:` (in-memory databases - limited backup support)

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
- Backup operations may be limited for in-memory databases (`:memory:`)
- The service automatically handles different SQLite URL formats
- Check logs for warnings if using non-standard database configurations
