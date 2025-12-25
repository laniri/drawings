"""
Backup and data persistence service for the drawing analysis system.
"""

import asyncio
import json
import logging
import shutil
import sqlite3
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiofiles

from app.core.config import settings
from app.core.exceptions import ConfigurationError, StorageError
from app.models.database import (
    AgeGroupModel,
    AnomalyAnalysis,
    Drawing,
    DrawingEmbedding,
    InterpretabilityResult,
)

logger = logging.getLogger(__name__)


class BackupService:
    """Service for managing database backups and data export/import."""

    def __init__(self, backup_dir: str = "backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)

        # Database path - handle different URL formats
        db_url = settings.DATABASE_URL
        if db_url.startswith("sqlite:///"):
            self.db_path = Path(db_url.replace("sqlite:///", ""))
        elif db_url.startswith("sqlite://"):
            # Handle in-memory databases or relative paths
            db_path_str = db_url.replace("sqlite://", "")
            if db_path_str == ":memory:":
                # For in-memory databases, we can't backup directly
                self.db_path = None
            else:
                self.db_path = Path(db_path_str)
        else:
            # Fallback for other database types
            self.db_path = Path("drawings.db")

        # Backup retention settings
        self.max_backup_age_days = 30
        self.max_backup_count = 10

        logger.info(f"BackupService initialized - Backup dir: {self.backup_dir}")
        if self.db_path:
            logger.info(f"Database path: {self.db_path}")
        else:
            logger.warning(
                "In-memory database detected - backup operations may be limited"
            )

    async def create_full_backup(self, include_files: bool = True) -> Dict[str, Any]:
        """
        Create a full system backup including database and files.

        Args:
            include_files: Whether to include uploaded files and generated content

        Returns:
            Backup information dictionary
        """
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_name = f"full_backup_{timestamp}"
            backup_path = self.backup_dir / f"{backup_name}.zip"

            logger.info(f"Creating full backup: {backup_name}")

            with zipfile.ZipFile(backup_path, "w", zipfile.ZIP_DEFLATED) as backup_zip:
                # Backup database
                if self.db_path.exists():
                    backup_zip.write(self.db_path, "database.db")
                    logger.info("Database added to backup")

                # Backup configuration (handle potential mock objects)
                try:
                    config_data = {
                        "PROJECT_NAME": getattr(settings, "PROJECT_NAME", "Unknown"),
                        "VERSION": getattr(settings, "VERSION", "1.0.0"),
                        "DATABASE_URL": getattr(
                            settings, "DATABASE_URL", "sqlite:///./drawings.db"
                        ),
                        "VISION_MODEL": getattr(
                            settings, "VISION_MODEL", "google/vit-base-patch16-224"
                        ),
                        "DEFAULT_THRESHOLD_PERCENTILE": getattr(
                            settings, "DEFAULT_THRESHOLD_PERCENTILE", 95.0
                        ),
                        "MIN_SAMPLES_PER_AGE_GROUP": getattr(
                            settings, "MIN_SAMPLES_PER_AGE_GROUP", 50
                        ),
                        "backup_timestamp": timestamp,
                        "backup_type": "full",
                    }

                    # Convert any non-serializable values to strings
                    serializable_config = {}
                    for key, value in config_data.items():
                        try:
                            json.dumps(value)  # Test if serializable
                            serializable_config[key] = value
                        except (TypeError, ValueError):
                            serializable_config[key] = str(value)

                    backup_zip.writestr(
                        "config.json", json.dumps(serializable_config, indent=2)
                    )
                    logger.info("Configuration added to backup")
                except Exception as e:
                    logger.warning(f"Could not backup configuration: {e}")
                    # Create minimal config
                    minimal_config = {
                        "backup_timestamp": timestamp,
                        "backup_type": "full",
                    }
                    backup_zip.writestr(
                        "config.json", json.dumps(minimal_config, indent=2)
                    )

                # Backup files if requested
                if include_files:
                    file_count = 0

                    # Backup uploaded files
                    upload_dir = Path(settings.UPLOAD_DIR)
                    if upload_dir.exists():
                        for file_path in upload_dir.rglob("*"):
                            if file_path.is_file():
                                arc_path = (
                                    f"uploads/{file_path.relative_to(upload_dir)}"
                                )
                                backup_zip.write(file_path, arc_path)
                                file_count += 1

                    # Backup static files (models, saliency maps, etc.)
                    static_dir = Path(settings.STATIC_DIR)
                    if static_dir.exists():
                        for file_path in static_dir.rglob("*"):
                            if file_path.is_file():
                                arc_path = f"static/{file_path.relative_to(static_dir)}"
                                backup_zip.write(file_path, arc_path)
                                file_count += 1

                    logger.info(f"Added {file_count} files to backup")

            backup_size = backup_path.stat().st_size

            # Clean up old backups
            await self._cleanup_old_backups()

            backup_info = {
                "backup_name": backup_name,
                "backup_path": str(backup_path),
                "timestamp": timestamp,
                "size_bytes": backup_size,
                "size_mb": round(backup_size / (1024 * 1024), 2),
                "includes_files": include_files,
                "status": "completed",
            }

            logger.info(f"Full backup completed: {backup_info}")
            return backup_info

        except Exception as e:
            logger.error(f"Full backup failed: {str(e)}")
            raise StorageError(f"Backup creation failed: {str(e)}")

    async def create_database_backup(self) -> Dict[str, Any]:
        """Create a database-only backup."""
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_name = f"db_backup_{timestamp}.db"
            backup_path = self.backup_dir / backup_name

            logger.info(f"Creating database backup: {backup_name}")

            if self.db_path is None:
                raise StorageError("Cannot backup in-memory database")

            if not self.db_path.exists():
                raise StorageError(f"Database file not found: {self.db_path}")

            # Ensure the database file is not locked by closing any connections
            # and copying with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Copy database file
                    shutil.copy2(self.db_path, backup_path)
                    break
                except (OSError, IOError) as e:
                    if attempt == max_retries - 1:
                        raise StorageError(
                            f"Failed to copy database after {max_retries} attempts: {str(e)}"
                        )
                    logger.warning(
                        f"Database copy attempt {attempt + 1} failed: {str(e)}, retrying..."
                    )
                    await asyncio.sleep(0.1)  # Brief delay before retry

            # Verify the backup was created and is not empty
            if not backup_path.exists():
                raise StorageError("Backup file was not created")

            backup_size = backup_path.stat().st_size
            if backup_size == 0:
                raise StorageError("Backup file is empty")

            backup_info = {
                "backup_name": backup_name,
                "backup_path": str(backup_path),
                "timestamp": timestamp,
                "size_bytes": backup_size,
                "size_mb": round(backup_size / (1024 * 1024), 2),
                "type": "database_only",
                "status": "completed",
            }

            logger.info(f"Database backup completed: {backup_info}")
            return backup_info

        except Exception as e:
            logger.error(f"Database backup failed: {str(e)}")
            raise StorageError(f"Database backup failed: {str(e)}")

    async def export_data(
        self, format: str = "json", include_embeddings: bool = False
    ) -> Dict[str, Any]:
        """
        Export system data in specified format.

        Args:
            format: Export format ("json", "csv")
            include_embeddings: Whether to include embedding vectors

        Returns:
            Export information dictionary
        """
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            export_name = f"data_export_{timestamp}"

            if format.lower() == "json":
                return await self._export_json(export_name, include_embeddings)
            elif format.lower() == "csv":
                return await self._export_csv(export_name, include_embeddings)
            else:
                raise ConfigurationError(f"Unsupported export format: {format}")

        except Exception as e:
            logger.error(f"Data export failed: {str(e)}")
            raise StorageError(f"Data export failed: {str(e)}")

    async def _export_json(
        self, export_name: str, include_embeddings: bool
    ) -> Dict[str, Any]:
        """Export data as JSON."""
        export_path = self.backup_dir / f"{export_name}.json"

        # Connect to database and extract data
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name

        export_data = {
            "export_info": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "format": "json",
                "includes_embeddings": include_embeddings,
            },
            "drawings": [],
            "age_group_models": [],
            "anomaly_analyses": [],
            "interpretability_results": [],
        }

        try:
            # Export drawings
            cursor = conn.execute("SELECT * FROM drawings")
            for row in cursor.fetchall():
                drawing_data = dict(row)
                export_data["drawings"].append(drawing_data)

            # Export embeddings if requested
            if include_embeddings:
                export_data["drawing_embeddings"] = []
                cursor = conn.execute("SELECT * FROM drawing_embeddings")
                for row in cursor.fetchall():
                    embedding_data = dict(row)
                    # Convert binary embedding to base64 for JSON serialization
                    if embedding_data["embedding_vector"]:
                        import base64

                        embedding_data["embedding_vector"] = base64.b64encode(
                            embedding_data["embedding_vector"]
                        ).decode("utf-8")
                    export_data["drawing_embeddings"].append(embedding_data)

            # Export age group models
            cursor = conn.execute("SELECT * FROM age_group_models")
            for row in cursor.fetchall():
                model_data = dict(row)
                export_data["age_group_models"].append(model_data)

            # Export analyses
            cursor = conn.execute("SELECT * FROM anomaly_analyses")
            for row in cursor.fetchall():
                analysis_data = dict(row)
                export_data["anomaly_analyses"].append(analysis_data)

            # Export interpretability results
            cursor = conn.execute("SELECT * FROM interpretability_results")
            for row in cursor.fetchall():
                interp_data = dict(row)
                export_data["interpretability_results"].append(interp_data)

        finally:
            conn.close()

        # Write JSON file
        async with aiofiles.open(export_path, "w") as f:
            await f.write(json.dumps(export_data, indent=2, default=str))

        export_size = export_path.stat().st_size

        return {
            "export_name": export_name,
            "export_path": str(export_path),
            "format": "json",
            "size_bytes": export_size,
            "size_mb": round(export_size / (1024 * 1024), 2),
            "record_counts": {
                "drawings": len(export_data["drawings"]),
                "age_group_models": len(export_data["age_group_models"]),
                "anomaly_analyses": len(export_data["anomaly_analyses"]),
                "interpretability_results": len(
                    export_data["interpretability_results"]
                ),
            },
            "includes_embeddings": include_embeddings,
            "status": "completed",
        }

    async def _export_csv(
        self, export_name: str, include_embeddings: bool
    ) -> Dict[str, Any]:
        """Export data as CSV files."""
        import csv

        export_dir = self.backup_dir / export_name
        export_dir.mkdir(exist_ok=True)

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        record_counts = {}

        try:
            # Export each table as a separate CSV
            tables = [
                "drawings",
                "age_group_models",
                "anomaly_analyses",
                "interpretability_results",
            ]

            if include_embeddings:
                tables.append("drawing_embeddings")

            for table in tables:
                csv_path = export_dir / f"{table}.csv"

                cursor = conn.execute(f"SELECT * FROM {table}")
                rows = cursor.fetchall()

                if rows:
                    # Write CSV file
                    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
                        writer.writeheader()

                        for row in rows:
                            row_dict = dict(row)
                            # Handle binary data for embeddings
                            if (
                                table == "drawing_embeddings"
                                and "embedding_vector" in row_dict
                            ):
                                if row_dict["embedding_vector"]:
                                    import base64

                                    row_dict["embedding_vector"] = base64.b64encode(
                                        row_dict["embedding_vector"]
                                    ).decode("utf-8")
                            writer.writerow(row_dict)

                record_counts[table] = len(rows)

        finally:
            conn.close()

        # Create zip archive of CSV files
        zip_path = self.backup_dir / f"{export_name}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for csv_file in export_dir.glob("*.csv"):
                zip_file.write(csv_file, csv_file.name)

        # Clean up temporary directory
        shutil.rmtree(export_dir)

        export_size = zip_path.stat().st_size

        return {
            "export_name": export_name,
            "export_path": str(zip_path),
            "format": "csv",
            "size_bytes": export_size,
            "size_mb": round(export_size / (1024 * 1024), 2),
            "record_counts": record_counts,
            "includes_embeddings": include_embeddings,
            "status": "completed",
        }

    async def restore_from_backup(
        self, backup_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Restore system from a backup file.

        Args:
            backup_path: Path to backup file

        Returns:
            Restore information dictionary
        """
        try:
            backup_path = Path(backup_path)

            if not backup_path.exists():
                raise StorageError(f"Backup file not found: {backup_path}")

            logger.info(f"Restoring from backup: {backup_path}")

            if self.db_path is None:
                raise StorageError("Cannot restore to in-memory database")

            # Create backup of current state before restore (if database exists)
            current_backup = None
            if self.db_path.exists():
                try:
                    current_backup = await self.create_database_backup()
                    logger.info(
                        f"Created safety backup: {current_backup['backup_name']}"
                    )
                except Exception as e:
                    logger.warning(f"Could not create safety backup: {e}")

            restore_info = {
                "backup_path": str(backup_path),
                "safety_backup": (
                    current_backup["backup_name"] if current_backup else None
                ),
                "restored_components": [],
                "status": "in_progress",
            }

            if backup_path.suffix == ".zip":
                # Full backup restore
                with zipfile.ZipFile(backup_path, "r") as backup_zip:
                    # Restore database
                    if "database.db" in backup_zip.namelist():
                        backup_zip.extract("database.db", self.backup_dir)
                        temp_db = self.backup_dir / "database.db"

                        # Ensure target directory exists
                        self.db_path.parent.mkdir(parents=True, exist_ok=True)

                        # Move with retry logic
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                shutil.move(temp_db, self.db_path)
                                break
                            except (OSError, IOError) as e:
                                if attempt == max_retries - 1:
                                    raise StorageError(
                                        f"Failed to restore database after {max_retries} attempts: {str(e)}"
                                    )
                                logger.warning(
                                    f"Database restore attempt {attempt + 1} failed: {str(e)}, retrying..."
                                )
                                await asyncio.sleep(0.1)

                        restore_info["restored_components"].append("database")
                        logger.info("Database restored")

                    # Restore files
                    for file_info in backup_zip.infolist():
                        if file_info.filename.startswith("uploads/"):
                            backup_zip.extract(file_info, Path("."))
                            restore_info["restored_components"].append("upload_files")
                        elif file_info.filename.startswith("static/"):
                            backup_zip.extract(file_info, Path("."))
                            restore_info["restored_components"].append("static_files")

            elif backup_path.suffix == ".db":
                # Database-only restore
                # Ensure target directory exists
                self.db_path.parent.mkdir(parents=True, exist_ok=True)

                # Verify backup file is valid before restore
                backup_size = backup_path.stat().st_size
                if backup_size == 0:
                    raise StorageError("Backup file is empty")

                # Test if backup file is a valid SQLite database
                try:
                    test_conn = sqlite3.connect(str(backup_path))
                    test_conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                    test_conn.close()
                except sqlite3.Error as e:
                    raise StorageError(
                        f"Backup file is not a valid SQLite database: {str(e)}"
                    )

                # Copy with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        shutil.copy2(backup_path, self.db_path)
                        break
                    except (OSError, IOError) as e:
                        if attempt == max_retries - 1:
                            raise StorageError(
                                f"Failed to restore database after {max_retries} attempts: {str(e)}"
                            )
                        logger.warning(
                            f"Database restore attempt {attempt + 1} failed: {str(e)}, retrying..."
                        )
                        await asyncio.sleep(0.1)

                # Verify the restored database
                if not self.db_path.exists():
                    raise StorageError("Database was not restored successfully")

                restored_size = self.db_path.stat().st_size
                if restored_size == 0:
                    raise StorageError("Restored database is empty")

                restore_info["restored_components"].append("database")
                logger.info("Database restored")

            else:
                raise ConfigurationError(
                    f"Unsupported backup format: {backup_path.suffix}"
                )

            restore_info["status"] = "completed"
            logger.info(f"Restore completed: {restore_info}")

            return restore_info

        except Exception as e:
            logger.error(f"Restore failed: {str(e)}")
            raise StorageError(f"Restore failed: {str(e)}")

    async def _cleanup_old_backups(self):
        """Clean up old backup files based on retention policy."""
        try:
            backup_files = list(self.backup_dir.glob("*backup*"))

            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Remove files older than max_backup_age_days
            cutoff_time = datetime.now().timestamp() - (
                self.max_backup_age_days * 24 * 3600
            )

            files_to_remove = []

            # Remove old files
            for backup_file in backup_files:
                if backup_file.stat().st_mtime < cutoff_time:
                    files_to_remove.append(backup_file)

            # Remove excess files (keep only max_backup_count newest)
            if len(backup_files) > self.max_backup_count:
                files_to_remove.extend(backup_files[self.max_backup_count :])

            # Remove duplicate files
            for file_to_remove in files_to_remove:
                if file_to_remove.exists():
                    file_to_remove.unlink()
                    logger.info(f"Removed old backup: {file_to_remove.name}")

            if files_to_remove:
                logger.info(f"Cleaned up {len(files_to_remove)} old backup files")

        except Exception as e:
            logger.warning(f"Backup cleanup failed: {str(e)}")

    async def get_backup_list(self) -> List[Dict[str, Any]]:
        """Get list of available backups."""
        try:
            backups = []

            for backup_file in self.backup_dir.glob("*backup*"):
                if backup_file.is_file():
                    stat = backup_file.stat()

                    backup_info = {
                        "name": backup_file.name,
                        "path": str(backup_file),
                        "size_bytes": stat.st_size,
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "type": "full" if backup_file.suffix == ".zip" else "database",
                    }

                    backups.append(backup_info)

            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x["created"], reverse=True)

            return backups

        except Exception as e:
            logger.error(f"Failed to get backup list: {str(e)}")
            raise StorageError(f"Failed to get backup list: {str(e)}")

    async def schedule_automatic_backup(self, interval_hours: int = 24):
        """Schedule automatic backups (basic implementation)."""
        logger.info(f"Scheduling automatic backups every {interval_hours} hours")

        while True:
            try:
                await asyncio.sleep(interval_hours * 3600)  # Convert hours to seconds

                logger.info("Running scheduled backup")
                backup_info = await self.create_database_backup()
                logger.info(f"Scheduled backup completed: {backup_info['backup_name']}")

            except Exception as e:
                logger.error(f"Scheduled backup failed: {str(e)}")
                # Continue the loop even if backup fails
                continue


# Global backup service instance
backup_service = BackupService()
