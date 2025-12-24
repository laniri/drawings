"""
Environment-aware file storage service.

This module provides a unified interface for file storage that automatically
switches between local storage and S3 based on the environment configuration.
"""

import os
import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
from datetime import datetime
from abc import ABC, abstractmethod
import aiofiles
from fastapi import UploadFile

from app.core.config import settings
from app.core.environment import StorageBackend
from app.core.exceptions import StorageError

logger = logging.getLogger(__name__)


class StorageBackendInterface(ABC):
    """Abstract interface for storage backends"""
    
    @abstractmethod
    async def save_uploaded_file(self, file: UploadFile, subdirectory: str = "drawings") -> Tuple[str, str]:
        """Save an uploaded file"""
        pass
    
    @abstractmethod
    async def save_file_from_bytes(self, file_data: bytes, filename: str, subdirectory: str = "generated") -> str:
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
            self.static_dir / "models"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _generate_unique_filename(self, original_filename: str, prefix: str = "") -> str:
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
    
    async def save_uploaded_file(self, file: UploadFile, subdirectory: str = "drawings") -> Tuple[str, str]:
        """Save an uploaded file to local storage"""
        try:
            unique_filename = self._generate_unique_filename(file.filename or "unknown.png")
            
            save_dir = self.upload_dir / subdirectory
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / unique_filename
            
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            await file.seek(0)
            
            logger.info(f"File saved to local storage: {file_path}")
            return unique_filename, str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save uploaded file to local storage: {str(e)}")
            raise StorageError(f"Local file save failed: {str(e)}")
    
    async def save_file_from_bytes(self, file_data: bytes, filename: str, subdirectory: str = "generated") -> str:
        """Save file data from bytes to local storage"""
        try:
            save_dir = self.static_dir / subdirectory
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / filename
            
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_data)
            
            logger.info(f"File saved from bytes to local storage: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save file from bytes to local storage: {str(e)}")
            raise StorageError(f"Local file save from bytes failed: {str(e)}")
    
    def get_file_url(self, file_path: str, base_url: str = "/static") -> str:
        """Generate a URL for accessing a locally stored file"""
        path = Path(file_path)
        
        if path.is_absolute():
            try:
                relative_path = path.relative_to(self.static_dir)
            except ValueError:
                relative_path = path.name
        else:
            relative_path = path
        
        url = f"{base_url}/{str(relative_path).replace(os.sep, '/')}"
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
            logger.error(f"Failed to delete file from local storage {file_path}: {str(e)}")
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
                "storage_backend": "local"
            }
        except Exception as e:
            logger.error(f"Failed to get file info from local storage for {file_path}: {str(e)}")
            return None


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
                self._s3_client = boto3.client('s3', region_name=self.aws_region)
            except ImportError:
                raise StorageError("boto3 is required for S3 storage backend")
        return self._s3_client
    
    def _generate_s3_key(self, filename: str, subdirectory: str) -> str:
        """Generate S3 object key"""
        return f"{subdirectory}/{filename}"
    
    def _generate_unique_filename(self, original_filename: str, prefix: str = "") -> str:
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
    
    async def save_uploaded_file(self, file: UploadFile, subdirectory: str = "drawings") -> Tuple[str, str]:
        """Save an uploaded file to S3"""
        try:
            unique_filename = self._generate_unique_filename(file.filename or "unknown.png")
            s3_key = self._generate_s3_key(unique_filename, subdirectory)
            
            # Read file content
            content = await file.read()
            await file.seek(0)
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=content,
                ContentType=file.content_type or 'application/octet-stream'
            )
            
            s3_url = f"s3://{self.bucket_name}/{s3_key}"
            logger.info(f"File saved to S3: {s3_url}")
            return unique_filename, s3_url
            
        except Exception as e:
            logger.error(f"Failed to save uploaded file to S3: {str(e)}")
            raise StorageError(f"S3 file save failed: {str(e)}")
    
    async def save_file_from_bytes(self, file_data: bytes, filename: str, subdirectory: str = "generated") -> str:
        """Save file data from bytes to S3"""
        try:
            s3_key = self._generate_s3_key(filename, subdirectory)
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=file_data,
                ContentType='application/octet-stream'
            )
            
            s3_url = f"s3://{self.bucket_name}/{s3_key}"
            logger.info(f"File saved from bytes to S3: {s3_url}")
            return s3_url
            
        except Exception as e:
            logger.error(f"Failed to save file from bytes to S3: {str(e)}")
            raise StorageError(f"S3 file save from bytes failed: {str(e)}")
    
    def get_file_url(self, file_path: str, base_url: str = "/static") -> str:
        """Generate a URL for accessing an S3 stored file"""
        if file_path.startswith("s3://"):
            # Extract S3 key from S3 URL
            s3_key = file_path.replace(f"s3://{self.bucket_name}/", "")
            
            # Generate presigned URL for secure access
            try:
                url = self.s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': self.bucket_name, 'Key': s3_key},
                    ExpiresIn=3600  # 1 hour
                )
                return url
            except Exception as e:
                logger.error(f"Failed to generate presigned URL for {file_path}: {str(e)}")
                return f"https://{self.bucket_name}.s3.{self.aws_region}.amazonaws.com/{s3_key}"
        
        # Fallback for non-S3 paths
        return f"{base_url}/{file_path}"
    
    def delete_file(self, file_path: str) -> bool:
        """Delete a file from S3"""
        try:
            if file_path.startswith("s3://"):
                s3_key = file_path.replace(f"s3://{self.bucket_name}/", "")
                
                self.s3_client.delete_object(
                    Bucket=self.bucket_name,
                    Key=s3_key
                )
                
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
            
            response = self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            return {
                "filename": Path(s3_key).name,
                "size": response['ContentLength'],
                "created": response.get('LastModified'),
                "modified": response.get('LastModified'),
                "extension": Path(s3_key).suffix.lower(),
                "exists": True,
                "storage_backend": "s3",
                "s3_key": s3_key,
                "content_type": response.get('ContentType')
            }
        except Exception as e:
            logger.error(f"Failed to get file info from S3 for {file_path}: {str(e)}")
            return None


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
                aws_region=env_config.aws_region or "eu-west-1"
            )
            logger.info(f"Initialized S3 storage backend: {env_config.s3_bucket_name}")
        else:
            self._backend = LocalStorageBackend(
                upload_dir=env_config.upload_dir,
                static_dir=env_config.static_dir
            )
            logger.info("Initialized local storage backend")
    
    @property
    def backend(self) -> StorageBackendInterface:
        """Get the current storage backend"""
        if self._backend is None:
            self._initialize_backend()
        return self._backend
    
    async def save_uploaded_file(self, file: UploadFile, subdirectory: str = "drawings") -> Tuple[str, str]:
        """Save an uploaded file using the appropriate backend"""
        return await self.backend.save_uploaded_file(file, subdirectory)
    
    async def save_file_from_bytes(self, file_data: bytes, filename: str, subdirectory: str = "generated") -> str:
        """Save file data from bytes using the appropriate backend"""
        return await self.backend.save_file_from_bytes(file_data, filename, subdirectory)
    
    def get_file_url(self, file_path: str, base_url: str = "/static") -> str:
        """Generate a URL for accessing a stored file"""
        return self.backend.get_file_url(file_path, base_url)
    
    def delete_file(self, file_path: str) -> bool:
        """Delete a file using the appropriate backend"""
        return self.backend.delete_file(file_path)
    
    def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get information about a stored file"""
        return self.backend.get_file_info(file_path)
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the current storage configuration"""
        env_config = settings.env_config
        return {
            "environment": env_config.environment.value,
            "storage_backend": env_config.storage_backend.value,
            "s3_bucket_name": env_config.s3_bucket_name,
            "aws_region": env_config.aws_region,
            "upload_dir": env_config.upload_dir,
            "static_dir": env_config.static_dir
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