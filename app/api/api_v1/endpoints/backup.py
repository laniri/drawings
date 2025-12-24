"""
Backup and data management endpoints.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.core.exceptions import ConfigurationError, StorageError
from app.services.backup_service import backup_service

router = APIRouter()


@router.post("/backup/full", summary="Create full system backup")
async def create_full_backup(
    include_files: bool = Query(
        default=True, description="Include uploaded files and generated content"
    )
):
    """Create a full system backup including database and files."""
    try:
        backup_info = await backup_service.create_full_backup(
            include_files=include_files
        )
        return backup_info

    except StorageError as e:
        raise HTTPException(status_code=507, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backup creation failed: {str(e)}")


@router.post("/backup/database", summary="Create database backup")
async def create_database_backup():
    """Create a database-only backup."""
    try:
        backup_info = await backup_service.create_database_backup()
        return backup_info

    except StorageError as e:
        raise HTTPException(status_code=507, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database backup failed: {str(e)}")


@router.get("/backup/list", summary="List available backups")
async def list_backups():
    """Get list of available backup files."""
    try:
        backups = await backup_service.get_backup_list()
        return {"backup_count": len(backups), "backups": backups}

    except StorageError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list backups: {str(e)}")


@router.get("/backup/download/{backup_name}", summary="Download backup file")
async def download_backup(backup_name: str):
    """Download a specific backup file."""
    try:
        backup_path = backup_service.backup_dir / backup_name

        if not backup_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Backup file not found: {backup_name}"
            )

        return FileResponse(
            path=str(backup_path),
            filename=backup_name,
            media_type="application/octet-stream",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@router.post("/backup/restore", summary="Restore from backup")
async def restore_from_backup(backup_name: str):
    """Restore system from a backup file."""
    try:
        backup_path = backup_service.backup_dir / backup_name

        if not backup_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Backup file not found: {backup_name}"
            )

        restore_info = await backup_service.restore_from_backup(backup_path)
        return restore_info

    except StorageError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Restore failed: {str(e)}")


@router.post("/backup/upload", summary="Upload backup file")
async def upload_backup(file: UploadFile = File(...)):
    """Upload a backup file for restoration."""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Validate file extension
        allowed_extensions = [".zip", ".db"]
        file_ext = Path(file.filename).suffix.lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}",
            )

        # Save uploaded file
        backup_path = backup_service.backup_dir / file.filename

        with open(backup_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        file_size = backup_path.stat().st_size

        return {
            "message": "Backup file uploaded successfully",
            "filename": file.filename,
            "size_bytes": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2),
            "path": str(backup_path),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/export", summary="Export system data")
async def export_data(
    format: str = Query(
        default="json", regex="^(json|csv)$", description="Export format"
    ),
    include_embeddings: bool = Query(
        default=False, description="Include embedding vectors"
    ),
):
    """Export system data in specified format."""
    try:
        export_info = await backup_service.export_data(
            format=format, include_embeddings=include_embeddings
        )
        return export_info

    except ConfigurationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except StorageError as e:
        raise HTTPException(status_code=507, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/export/download/{export_name}", summary="Download exported data")
async def download_export(export_name: str):
    """Download an exported data file."""
    try:
        # Try different extensions
        for ext in [".json", ".zip"]:
            export_path = backup_service.backup_dir / f"{export_name}{ext}"
            if export_path.exists():
                return FileResponse(
                    path=str(export_path),
                    filename=f"{export_name}{ext}",
                    media_type="application/octet-stream",
                )

        raise HTTPException(
            status_code=404, detail=f"Export file not found: {export_name}"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@router.delete("/backup/{backup_name}", summary="Delete backup file")
async def delete_backup(backup_name: str):
    """Delete a specific backup file."""
    try:
        backup_path = backup_service.backup_dir / backup_name

        if not backup_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Backup file not found: {backup_name}"
            )

        # Get file info before deletion
        file_size = backup_path.stat().st_size

        # Delete the file
        backup_path.unlink()

        return {
            "message": f"Backup file deleted successfully",
            "filename": backup_name,
            "size_freed_mb": round(file_size / (1024 * 1024), 2),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


@router.post("/cleanup", summary="Clean up old backups")
async def cleanup_backups():
    """Clean up old backup files based on retention policy."""
    try:
        # Get current backup list
        backups_before = await backup_service.get_backup_list()

        # Run cleanup
        await backup_service._cleanup_old_backups()

        # Get updated backup list
        backups_after = await backup_service.get_backup_list()

        cleaned_count = len(backups_before) - len(backups_after)

        return {
            "message": "Backup cleanup completed",
            "backups_before": len(backups_before),
            "backups_after": len(backups_after),
            "files_cleaned": cleaned_count,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@router.get("/storage/info", summary="Get storage information")
async def get_storage_info():
    """Get information about storage usage and organization."""
    try:
        from pathlib import Path

        import psutil

        # Get disk usage
        disk_usage = psutil.disk_usage("/")

        # Get directory sizes
        def get_dir_size(path: Path) -> int:
            if not path.exists():
                return 0
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

        upload_dir = backup_service.upload_dir
        static_dir = backup_service.static_dir
        backup_dir = backup_service.backup_dir

        storage_info = {
            "disk_usage": {
                "total_gb": round(disk_usage.total / (1024**3), 2),
                "used_gb": round(disk_usage.used / (1024**3), 2),
                "free_gb": round(disk_usage.free / (1024**3), 2),
                "used_percent": round((disk_usage.used / disk_usage.total) * 100, 1),
            },
            "directories": {
                "uploads": {
                    "path": str(upload_dir),
                    "exists": upload_dir.exists(),
                    "size_mb": (
                        round(get_dir_size(upload_dir) / (1024**2), 2)
                        if upload_dir.exists()
                        else 0
                    ),
                    "file_count": (
                        len(list(upload_dir.rglob("*"))) if upload_dir.exists() else 0
                    ),
                },
                "static": {
                    "path": str(static_dir),
                    "exists": static_dir.exists(),
                    "size_mb": (
                        round(get_dir_size(static_dir) / (1024**2), 2)
                        if static_dir.exists()
                        else 0
                    ),
                    "file_count": (
                        len(list(static_dir.rglob("*"))) if static_dir.exists() else 0
                    ),
                },
                "backups": {
                    "path": str(backup_dir),
                    "exists": backup_dir.exists(),
                    "size_mb": (
                        round(get_dir_size(backup_dir) / (1024**2), 2)
                        if backup_dir.exists()
                        else 0
                    ),
                    "file_count": (
                        len(list(backup_dir.rglob("*"))) if backup_dir.exists() else 0
                    ),
                },
            },
            "database": {
                "path": str(backup_service.db_path),
                "exists": backup_service.db_path.exists(),
                "size_mb": (
                    round(backup_service.db_path.stat().st_size / (1024**2), 2)
                    if backup_service.db_path.exists()
                    else 0
                ),
            },
        }

        return storage_info

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Storage info retrieval failed: {str(e)}"
        )


@router.post("/storage/cleanup", summary="Clean up temporary and orphaned files")
async def cleanup_storage():
    """Clean up temporary files and orphaned data."""
    try:
        cleanup_results = {
            "temp_files_removed": 0,
            "orphaned_files_removed": 0,
            "space_freed_mb": 0,
        }

        # Clean up temporary files
        temp_patterns = ["*.tmp", "*.temp", "*~", ".DS_Store"]

        for pattern in temp_patterns:
            for temp_file in Path(".").rglob(pattern):
                if temp_file.is_file():
                    file_size = temp_file.stat().st_size
                    temp_file.unlink()
                    cleanup_results["temp_files_removed"] += 1
                    cleanup_results["space_freed_mb"] += file_size / (1024**2)

        cleanup_results["space_freed_mb"] = round(cleanup_results["space_freed_mb"], 2)

        return {"message": "Storage cleanup completed", **cleanup_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage cleanup failed: {str(e)}")
