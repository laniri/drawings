"""
File storage service for handling uploaded drawings and generated files.

This module provides functionality for storing, organizing, and managing
uploaded drawing files and generated analysis results.
"""

import logging
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import aiofiles
from fastapi import UploadFile

from app.core.config import settings
from app.core.exceptions import StorageError

logger = logging.getLogger(__name__)


# Keep the old exception for backward compatibility in tests
class FileStorageError(StorageError):
    """Custom exception for file storage operations (deprecated, use StorageError)"""

    pass


class FileStorageService:
    """Service for managing file storage operations"""

    def __init__(self, base_upload_dir: str = None, base_static_dir: str = None):
        """
        Initialize the file storage service

        Args:
            base_upload_dir: Base directory for uploaded files
            base_static_dir: Base directory for static files
        """
        self.upload_dir = Path(base_upload_dir or settings.UPLOAD_DIR)
        self.static_dir = Path(base_static_dir or settings.STATIC_DIR)

        # Create directories if they don't exist
        self._ensure_directories()

        logger.info(
            f"FileStorageService initialized - Upload: {self.upload_dir}, Static: {self.static_dir}"
        )

    def _ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.upload_dir,
            self.static_dir,
            self.upload_dir / "drawings",
            self.static_dir / "saliency_maps",
            self.static_dir / "overlays",
            self.static_dir / "models",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")

    def generate_unique_filename(self, original_filename: str, prefix: str = "") -> str:
        """
        Generate a unique filename while preserving the original extension

        Args:
            original_filename: Original filename from upload
            prefix: Optional prefix for the filename

        Returns:
            Unique filename with timestamp and UUID
        """
        # Extract file extension
        file_path = Path(original_filename)
        extension = file_path.suffix.lower()

        # Generate unique identifier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]

        # Construct filename
        if prefix:
            filename = f"{prefix}_{timestamp}_{unique_id}{extension}"
        else:
            filename = f"{timestamp}_{unique_id}{extension}"

        return filename

    async def save_uploaded_file(
        self, file: UploadFile, subdirectory: str = "drawings"
    ) -> Tuple[str, str]:
        """
        Save an uploaded file to the storage system

        Args:
            file: FastAPI UploadFile object
            subdirectory: Subdirectory within upload_dir to save the file

        Returns:
            Tuple of (filename, full_file_path)

        Raises:
            FileStorageError: If file saving fails
        """
        try:
            # Generate unique filename
            unique_filename = self.generate_unique_filename(
                file.filename or "unknown.png"
            )

            # Determine save path
            save_dir = self.upload_dir / subdirectory
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / unique_filename

            # Save file asynchronously
            async with aiofiles.open(file_path, "wb") as f:
                content = await file.read()
                await f.write(content)

            # Reset file position for potential reuse
            await file.seek(0)

            logger.info(f"File saved successfully: {file_path}")
            return unique_filename, str(file_path)

        except Exception as e:
            logger.error(f"Failed to save uploaded file: {str(e)}")
            raise StorageError(f"File save failed: {str(e)}")

    async def save_file_from_bytes(
        self, file_data: bytes, filename: str, subdirectory: str = "generated"
    ) -> str:
        """
        Save file data from bytes to the storage system

        Args:
            file_data: Raw file bytes
            filename: Desired filename
            subdirectory: Subdirectory within static_dir to save the file

        Returns:
            Full file path where the file was saved

        Raises:
            FileStorageError: If file saving fails
        """
        try:
            # Determine save path
            save_dir = self.static_dir / subdirectory
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / filename

            # Save file asynchronously
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(file_data)

            logger.info(f"File saved from bytes: {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to save file from bytes: {str(e)}")
            raise StorageError(f"File save from bytes failed: {str(e)}")

    def get_file_url(self, file_path: str, base_url: str = "/static") -> str:
        """
        Generate a URL for accessing a stored file

        Args:
            file_path: Full file path or relative path from static directory
            base_url: Base URL for static files

        Returns:
            URL for accessing the file
        """
        # Convert to Path object for easier manipulation
        path = Path(file_path)

        # If it's an absolute path, make it relative to static_dir
        if path.is_absolute():
            try:
                relative_path = path.relative_to(self.static_dir)
            except ValueError:
                # Path is not under static_dir, use filename only
                relative_path = path.name
        else:
            relative_path = path

        # Construct URL with forward slashes
        url = f"{base_url}/{str(relative_path).replace(os.sep, '/')}"
        return url

    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from the storage system

        Args:
            file_path: Path to the file to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            path = Path(file_path)
            if path.exists() and path.is_file():
                path.unlink()
                logger.info(f"File deleted successfully: {file_path}")
                return True
            else:
                logger.warning(f"File not found for deletion: {file_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {str(e)}")
            return False

    def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a stored file

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with file information or None if file doesn't exist
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return None

            stat = path.stat()
            return {
                "filename": path.name,
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime),
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "extension": path.suffix.lower(),
                "exists": True,
            }
        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {str(e)}")
            return None

    def cleanup_old_files(self, directory: str, max_age_days: int = 30) -> int:
        """
        Clean up old files from a directory

        Args:
            directory: Directory to clean up
            max_age_days: Maximum age of files to keep in days

        Returns:
            Number of files deleted
        """
        try:
            cleanup_dir = Path(directory)
            if not cleanup_dir.exists():
                return 0

            cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
            deleted_count = 0

            for file_path in cleanup_dir.rglob("*"):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted old file: {file_path}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to delete old file {file_path}: {str(e)}"
                        )

            logger.info(
                f"Cleanup completed: {deleted_count} files deleted from {directory}"
            )
            return deleted_count

        except Exception as e:
            logger.error(f"Cleanup failed for directory {directory}: {str(e)}")
            return 0

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics

        Returns:
            Dictionary with storage statistics
        """
        try:
            stats = {
                "upload_dir": str(self.upload_dir),
                "static_dir": str(self.static_dir),
                "directories": {},
            }

            # Check each main directory
            for dir_name, dir_path in [
                ("uploads", self.upload_dir),
                ("static", self.static_dir),
            ]:
                if dir_path.exists():
                    file_count = sum(1 for _ in dir_path.rglob("*") if _.is_file())
                    total_size = sum(
                        f.stat().st_size for f in dir_path.rglob("*") if f.is_file()
                    )

                    stats["directories"][dir_name] = {
                        "exists": True,
                        "file_count": file_count,
                        "total_size_bytes": total_size,
                        "total_size_mb": round(total_size / (1024 * 1024), 2),
                    }
                else:
                    stats["directories"][dir_name] = {"exists": False}

            return stats

        except Exception as e:
            logger.error(f"Failed to get storage stats: {str(e)}")
            return {"error": str(e)}
