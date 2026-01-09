"""
Environment-aware file storage service.

This module provides a unified interface for file storage that automatically
switches between local storage and S3 based on the environment configuration.
"""

import logging
import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import aiofiles
from fastapi import UploadFile

from app.core.config import settings
from app.core.environment import StorageBackend
from app.core.exceptions import StorageError

logger = logging.getLogger(__name__)


class StorageBackendInterface(ABC):
    """Abstract interface for storage backends"""

    @abstractmethod
    async def save_uploaded_file(
        self, file: UploadFile, subdirectory: str = "drawings"
    ) -> Tuple[str, str]:
        """Save an uploaded file"""
        pass

    @abstractmethod
    async def save_file_from_bytes(
        self, file_data: bytes, filename: str, subdirectory: str = "generated"
    ) -> str:
        """Save file data from bytes"""
        pass

    @abstractmethod
    def get_file_url(self, file_path: str, base_url: str = "/static") -> str:
        """Generate a URL for accessing a stored file"""
        pass

    @abstractmethod
    def delete_file(self, file_path: str) -> bool:
        """Delete a file"""
        pass

    @abstractmethod
    def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get information about a stored file"""
        pass


class LocalStorageBackend(StorageBackendInterface):
    """Local file system storage backend"""

    def __init__(self, upload_dir: str, static_dir: str):
        self.upload_dir = Path(upload_dir)
        self.static_dir = Path(static_dir)
        self._ensure_directories()

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

    def _generate_unique_filename(
        self, original_filename: str, prefix: str = ""
    ) -> str:
        """Generate a unique filename while preserving the original extension"""
        file_path = Path(original_filename)
        extension = file_path.suffix.lower()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]

        if prefix:
            filename = f"{prefix}_{timestamp}_{unique_id}{extension}"
        else:
            filename = f"{timestamp}_{unique_id}{extension}"

        return filename

    async def save_uploaded_file(
        self, file: UploadFile, subdirectory: str = "drawings"
    ) -> Tuple[str, str]:
        """Save an uploaded file to local storage"""
        try:
            unique_filename = self._generate_unique_filename(
                file.filename or "unknown.png"
            )

            save_dir = self.upload_dir / subdirectory
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / unique_filename

            async with aiofiles.open(file_path, "wb") as f:
                content = await file.read()
                await f.write(content)

            await file.seek(0)

            logger.info(f"File saved to local storage: {file_path}")
            return unique_filename, str(file_path)

        except Exception as e:
            logger.error(f"Failed to save uploaded file to local storage: {str(e)}")
            raise StorageError(f"Local file save failed: {str(e)}")

    async def save_file_from_bytes(
        self, file_data: bytes, filename: str, subdirectory: str = "generated"
    ) -> str:
        """Save file data from bytes to local storage"""
        try:
            save_dir = self.static_dir / subdirectory
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / filename

            async with aiofiles.open(file_path, "wb") as f:
                await f.write(file_data)

            logger.info(f"File saved from bytes to local storage: {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to save file from bytes to local storage: {str(e)}")
            raise StorageError(f"Local file save from bytes failed: {str(e)}")

    def get_file_url(self, file_path: str, base_url: str = "/static") -> str:
        """Generate a URL for accessing a locally stored file"""
        path = Path(file_path)

        # Handle absolute paths
        if path.is_absolute():
            # Try to make relative to static_dir
            try:
                relative_path = path.relative_to(self.static_dir)
                url = f"/static/{str(relative_path).replace(os.sep, '/')}"
                logger.debug(f"Local URL (absolute->static): {file_path} -> {url}")
                return url
            except ValueError:
                # Try to make relative to upload_dir
                try:
                    relative_path = path.relative_to(self.upload_dir)
                    url = f"/uploads/{str(relative_path).replace(os.sep, '/')}"
                    logger.debug(f"Local URL (absolute->uploads): {file_path} -> {url}")
                    return url
                except ValueError:
                    # Can't determine base, use filename only
                    url = f"{base_url}/{path.name}"
                    logger.warning(f"Local URL (fallback): {file_path} -> {url}")
                    return url

        # Handle relative paths - they already include the directory structure
        path_str = str(path).replace(os.sep, "/")

        # If path already starts with uploads/ or static/, just prepend /
        if path_str.startswith("uploads/"):
            url = f"/{path_str}"
            logger.debug(f"Local URL (relative uploads): {file_path} -> {url}")
            return url
        elif path_str.startswith("static/"):
            url = f"/{path_str}"
            logger.debug(f"Local URL (relative static): {file_path} -> {url}")
            return url
        else:
            # Otherwise use base_url
            url = f"{base_url}/{path_str}"
            logger.debug(f"Local URL (base_url): {file_path} -> {url}")
            return url

    def delete_file(self, file_path: str) -> bool:
        """Delete a file from local storage"""
        try:
            path = Path(file_path)
            if path.exists() and path.is_file():
                path.unlink()
                logger.info(f"File deleted from local storage: {file_path}")
                return True
            else:
                logger.warning(f"File not found for deletion: {file_path}")
                return False
        except Exception as e:
            logger.error(
                f"Failed to delete file from local storage {file_path}: {str(e)}"
            )
            return False

    def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get information about a locally stored file"""
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
                "storage_backend": "local",
            }
        except Exception as e:
            logger.error(
                f"Failed to get file info from local storage for {file_path}: {str(e)}"
            )
            return None

    def download_to_local(self, file_path: str) -> str:
        """For local storage, just return the path as-is"""
        return file_path


class S3StorageBackend(StorageBackendInterface):
    """AWS S3 storage backend"""

    def __init__(self, bucket_name: str, aws_region: str):
        self.bucket_name = bucket_name
        self.aws_region = aws_region
        self._s3_client = None

        # Local fallback directories for temporary operations
        self.temp_upload_dir = Path("temp_uploads")
        self.temp_static_dir = Path("temp_static")
        self._ensure_temp_directories()

    def _ensure_temp_directories(self):
        """Ensure temporary directories exist for local operations"""
        self.temp_upload_dir.mkdir(exist_ok=True)
        self.temp_static_dir.mkdir(exist_ok=True)

    @property
    def s3_client(self):
        """Lazy initialization of S3 client"""
        if self._s3_client is None:
            try:
                import boto3

                self._s3_client = boto3.client("s3", region_name=self.aws_region)
            except ImportError:
                raise StorageError("boto3 is required for S3 storage backend")
        return self._s3_client

    def _generate_s3_key(self, filename: str, subdirectory: str) -> str:
        """Generate S3 object key"""
        return f"{subdirectory}/{filename}"

    def _generate_unique_filename(
        self, original_filename: str, prefix: str = ""
    ) -> str:
        """Generate a unique filename while preserving the original extension"""
        file_path = Path(original_filename)
        extension = file_path.suffix.lower()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]

        if prefix:
            filename = f"{prefix}_{timestamp}_{unique_id}{extension}"
        else:
            filename = f"{timestamp}_{unique_id}{extension}"

        return filename

    async def save_uploaded_file(
        self, file: UploadFile, subdirectory: str = "drawings"
    ) -> Tuple[str, str]:
        """Save an uploaded file to S3"""
        try:
            unique_filename = self._generate_unique_filename(
                file.filename or "unknown.png"
            )
            s3_key = self._generate_s3_key(unique_filename, subdirectory)

            # Read file content
            content = await file.read()
            await file.seek(0)

            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=content,
                ContentType=file.content_type or "application/octet-stream",
            )

            s3_url = f"s3://{self.bucket_name}/{s3_key}"
            logger.info(f"File saved to S3: {s3_url}")
            return unique_filename, s3_url

        except Exception as e:
            logger.error(f"Failed to save uploaded file to S3: {str(e)}")
            raise StorageError(f"S3 file save failed: {str(e)}")

    async def save_file_from_bytes(
        self, file_data: bytes, filename: str, subdirectory: str = "generated"
    ) -> str:
        """Save file data from bytes to S3"""
        try:
            s3_key = self._generate_s3_key(filename, subdirectory)

            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=file_data,
                ContentType="application/octet-stream",
            )

            s3_url = f"s3://{self.bucket_name}/{s3_key}"
            logger.info(f"File saved from bytes to S3: {s3_url}")
            return s3_url

        except Exception as e:
            logger.error(f"Failed to save file from bytes to S3: {str(e)}")
            raise StorageError(f"S3 file save from bytes failed: {str(e)}")

    def get_file_url(self, file_path: str, base_url: str = "/static") -> str:
        """Generate a URL for accessing an S3 stored file"""
        logger.info(f"[S3 URL Generation] Input path: {file_path}")
        
        # Handle relative paths that are stored in database (uploads/..., static/...)
        if not file_path.startswith("s3://"):
            # These are relative paths stored in DB - treat them as local paths
            # that should be served directly by nginx
            path_str = file_path.replace(os.sep, "/")
            logger.info(f"[S3 URL Generation] Normalized path: {path_str}")

            # Drawings in uploads/ are synced during startup - serve directly
            if path_str.startswith("uploads/"):
                url = f"/{path_str}"
                logger.info(f"[S3 URL Generation] Drawing (synced): {file_path} -> {url}")
                return url
            # Saliency maps in static/saliency_maps/ are NOT synced (too many files)
            # Must be served via API endpoint for on-demand S3 fetch
            # S3 bucket structure: static/saliency_maps/ (WITH static/ prefix)
            elif path_str.startswith("static/saliency_maps/"):
                # Keep the full path including static/ for S3 key
                url = f"/api/v1/files/s3/{path_str}"
                logger.info(f"[S3 URL Generation] Saliency (on-demand): {file_path} -> {url}")
                logger.info(f"[S3 URL Generation] Will fetch from S3: s3://{self.bucket_name}/{path_str}")
                return url
            # Other static/ files might be synced - serve directly
            elif path_str.startswith("static/"):
                url = f"/{path_str}"
                logger.info(f"[S3 URL Generation] Static (synced): {file_path} -> {url}")
                return url
            else:
                # Otherwise use base_url
                url = f"{base_url}/{path_str}"
                logger.info(f"[S3 URL Generation] Fallback: {file_path} -> {url}")
                return url

        # Handle S3 URLs (s3://bucket/key)
        # Extract S3 key from S3 URL
        s3_key = file_path.replace(f"s3://{self.bucket_name}/", "")

        # Check if file type is synced during container startup
        # Only drawings/uploads are synced; saliency maps are NOT synced (too many files)
        local_path = self._get_local_path_for_s3_key(s3_key)
        if local_path and s3_key.startswith("drawings/"):
            # Drawings are synced to /app/uploads/ during startup
            # Check both relative path and absolute path (for container)
            absolute_path = Path("/app") / local_path
            if Path(local_path).exists() or absolute_path.exists():
                # File exists locally, return local URL
                # nginx will serve it directly
                url = f"/{local_path}"
                logger.debug(f"S3 URL (synced local): {file_path} -> {url}")
                return url

        # File not synced or doesn't exist locally
        # Return API endpoint that will download it on-demand
        url = f"/api/v1/files/s3/{s3_key}"
        logger.debug(f"S3 URL (API endpoint): {file_path} -> {url}")
        return url

    def _get_local_path_for_s3_key(self, s3_key: str) -> Optional[str]:
        """
        Map S3 key to local file path.

        During container startup, only drawings are synced from S3:
        - s3://bucket/drawings/* -> /app/uploads/

        Saliency maps are NOT synced (too many files) and must be fetched on-demand.

        Args:
            s3_key: S3 object key (e.g., "drawings/file.png")

        Returns:
            Local file path if mapping exists, None otherwise
        """
        if s3_key.startswith("drawings/"):
            # Drawing files are synced to uploads/
            filename = s3_key.replace("drawings/", "")
            return f"uploads/{filename}"
        elif s3_key.startswith("saliency_maps/"):
            # Saliency maps are NOT synced - return path for reference only
            # Caller should check if file exists before using
            return f"static/{s3_key}"
        elif s3_key.startswith("overlays/"):
            # Overlay images are NOT synced - return path for reference only
            return f"static/{s3_key}"

        # Unknown S3 key pattern
        return None

    def delete_file(self, file_path: str) -> bool:
        """Delete a file from S3"""
        try:
            if file_path.startswith("s3://"):
                s3_key = file_path.replace(f"s3://{self.bucket_name}/", "")

                self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)

                logger.info(f"File deleted from S3: {file_path}")
                return True
            else:
                logger.warning(f"Invalid S3 path for deletion: {file_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete file from S3 {file_path}: {str(e)}")
            return False

    def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get information about an S3 stored file"""
        try:
            if not file_path.startswith("s3://"):
                return None

            s3_key = file_path.replace(f"s3://{self.bucket_name}/", "")

            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)

            return {
                "filename": Path(s3_key).name,
                "size": response["ContentLength"],
                "created": response.get("LastModified"),
                "modified": response.get("LastModified"),
                "extension": Path(s3_key).suffix.lower(),
                "exists": True,
                "storage_backend": "s3",
                "s3_key": s3_key,
                "content_type": response.get("ContentType"),
            }
        except Exception as e:
            logger.error(f"Failed to get file info from S3 for {file_path}: {str(e)}")
            return None

    def download_to_local(self, file_path: str) -> str:
        """Download an S3 file to a temporary local path for processing.

        Optimized to check if file was already synced locally before downloading.
        """
        try:
            if not file_path.startswith("s3://"):
                # Already a local path
                return file_path

            s3_key = file_path.replace(f"s3://{self.bucket_name}/", "")
            filename = Path(s3_key).name

            # Check if file was already synced to local uploads directory
            # The background sync copies files to /app/uploads/
            if s3_key.startswith("uploads/") or s3_key.startswith("drawings/"):
                # Try local path first (from background sync)
                local_synced_path = Path("/app") / s3_key
                if local_synced_path.exists():
                    logger.debug(f"Using locally synced file: {local_synced_path}")
                    return str(local_synced_path)

                # Also check uploads directory directly
                uploads_path = Path("/app/uploads") / filename
                if uploads_path.exists():
                    logger.debug(f"Using locally synced file: {uploads_path}")
                    return str(uploads_path)

            # Create temporary local path for download
            local_path = self.temp_upload_dir / filename

            # Check if already downloaded to temp
            if local_path.exists():
                logger.debug(f"Using cached temp file: {local_path}")
                return str(local_path)

            # Download from S3
            self.s3_client.download_file(
                Bucket=self.bucket_name, Key=s3_key, Filename=str(local_path)
            )

            logger.info(f"Downloaded S3 file to local: {file_path} -> {local_path}")
            return str(local_path)

        except Exception as e:
            logger.error(f"Failed to download S3 file {file_path}: {str(e)}")
            raise StorageError(f"S3 file download failed: {str(e)}")


class EnvironmentAwareStorageService:
    """
    Environment-aware storage service that automatically switches between
    local and S3 storage based on environment configuration.
    """

    def __init__(self):
        """Initialize storage service with environment-appropriate backend"""
        self._backend = None
        self._initialize_backend()

    def _initialize_backend(self):
        """Initialize the appropriate storage backend based on environment"""
        env_config = settings.env_config

        if env_config.storage_backend == StorageBackend.S3:
            if not env_config.s3_bucket_name:
                raise StorageError("S3 bucket name is required for S3 storage backend")

            self._backend = S3StorageBackend(
                bucket_name=env_config.s3_bucket_name,
                aws_region=env_config.aws_region or "eu-west-1",
            )
            logger.info(f"Initialized S3 storage backend: {env_config.s3_bucket_name}")
        else:
            self._backend = LocalStorageBackend(
                upload_dir=env_config.upload_dir, static_dir=env_config.static_dir
            )
            logger.info("Initialized local storage backend")

    @property
    def backend(self) -> StorageBackendInterface:
        """Get the current storage backend"""
        if self._backend is None:
            self._initialize_backend()
        return self._backend

    async def save_uploaded_file(
        self, file: UploadFile, subdirectory: str = "drawings"
    ) -> Tuple[str, str]:
        """Save an uploaded file using the appropriate backend"""
        return await self.backend.save_uploaded_file(file, subdirectory)

    async def save_file_from_bytes(
        self, file_data: bytes, filename: str, subdirectory: str = "generated"
    ) -> str:
        """Save file data from bytes using the appropriate backend"""
        return await self.backend.save_file_from_bytes(
            file_data, filename, subdirectory
        )

    def get_file_url(self, file_path: str, base_url: str = "/static") -> str:
        """Generate a URL for accessing a stored file"""
        return self.backend.get_file_url(file_path, base_url)

    def delete_file(self, file_path: str) -> bool:
        """Delete a file using the appropriate backend"""
        return self.backend.delete_file(file_path)

    def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get information about a stored file"""
        return self.backend.get_file_info(file_path)

    def download_to_local(self, file_path: str) -> str:
        """Download a file to local path for processing (handles both S3 and local paths)"""
        return self.backend.download_to_local(file_path)

    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the current storage configuration"""
        env_config = settings.env_config
        return {
            "environment": env_config.environment.value,
            "storage_backend": env_config.storage_backend.value,
            "s3_bucket_name": env_config.s3_bucket_name,
            "aws_region": env_config.aws_region,
            "upload_dir": env_config.upload_dir,
            "static_dir": env_config.static_dir,
        }


# Global storage service instance
_storage_service: Optional[EnvironmentAwareStorageService] = None


def get_storage_service() -> EnvironmentAwareStorageService:
    """
    Get or create the global storage service instance.

    Returns:
        EnvironmentAwareStorageService: Global storage service
    """
    global _storage_service
    if _storage_service is None:
        _storage_service = EnvironmentAwareStorageService()
    return _storage_service


def reset_storage_service():
    """
    Reset the global storage service.

    This is primarily useful for testing to force re-initialization
    of the storage service.
    """
    global _storage_service
    _storage_service = None
