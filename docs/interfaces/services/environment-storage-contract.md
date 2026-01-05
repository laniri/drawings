# Environment-Aware Storage Contract

## Overview
Service contract for Environment-Aware Storage Service

**Source File**: `app/services/environment_storage.py`

## Interface Specification

### Classes

#### EnvironmentAwareStorageService

Service that automatically switches between local and S3 storage based on environment configuration.

**Key Features**:
- Automatic backend selection (local filesystem or AWS S3)
- Environment-based configuration
- Unified interface for all storage operations
- Presigned URL generation for S3 access
- Graceful fallback handling

#### StorageBackendInterface

Abstract interface for storage backends

**Implemented by**:
- `LocalStorageBackend` - Local filesystem storage
- `S3StorageBackend` - AWS S3 storage

## Methods

### save_uploaded_file

Save an uploaded file using the appropriate backend

**Signature**: `save_uploaded_file(file: UploadFile, subdirectory: str = "drawings") -> Tuple[str, str]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `file` | `UploadFile` | FastAPI uploaded file object |
| `subdirectory` | `str` | Subdirectory for file organization (default: "drawings") |

**Returns**: `Tuple[str, str]` - (unique_filename, full_file_path)

### save_file_from_bytes

Save file data from bytes using the appropriate backend

**Signature**: `save_file_from_bytes(file_data: bytes, filename: str, subdirectory: str = "generated") -> str`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `file_data` | `bytes` | Raw file data |
| `filename` | `str` | Desired filename |
| `subdirectory` | `str` | Subdirectory for file organization (default: "generated") |

**Returns**: `str` - Full file path

### get_file_url

Generate a URL for accessing a stored file (environment-aware)

**Behavior**:
- **Local environment**: Returns local file URL (e.g., `/static/file.png`)
- **Production environment**: Returns S3 presigned URL (expires in 1 hour)

**Signature**: `get_file_url(file_path: str, base_url: str = "/static") -> str`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `file_path` | `str` | File path (absolute or relative) |
| `base_url` | `str` | Base URL for local files (default: "/static") |

**Returns**: `str` - Accessible URL for the file

### delete_file

Delete a file using the appropriate backend

**Signature**: `delete_file(file_path: str) -> bool`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `file_path` | `str` | Path to the file to delete |

**Returns**: `bool` - True if deletion was successful

### get_file_info

Get information about a stored file

**Signature**: `get_file_info(file_path: str) -> Optional[Dict[str, Any]]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `file_path` | `str` | Path to the file |

**Returns**: `Optional[Dict[str, Any]]` - File information or None if not found

### get_storage_info

Get information about the current storage configuration

**Signature**: `get_storage_info() -> Dict[str, Any]`

**Returns**: `Dict[str, Any]` - Storage configuration details

**Example Response**:
```json
{
  "environment": "production",
  "storage_backend": "s3",
  "s3_bucket_name": "my-app-bucket",
  "aws_region": "eu-west-1",
  "upload_dir": "uploads",
  "static_dir": "static"
}
```

## Environment Configuration

### Local Environment
- **Storage Backend**: Local filesystem
- **File URLs**: Local paths (e.g., `/static/saliency_maps/file.png`)
- **Configuration**: Uses `UPLOAD_DIR` and `STATIC_DIR` settings

### Production Environment
- **Storage Backend**: AWS S3
- **File URLs**: Presigned URLs with 1-hour expiration
- **Configuration**: Uses `S3_BUCKET_NAME` and `AWS_REGION` settings

## Global Service Access

### get_storage_service()

Get or create the global storage service instance.

**Signature**: `get_storage_service() -> EnvironmentAwareStorageService`

**Returns**: `EnvironmentAwareStorageService` - Global storage service instance

### reset_storage_service()

Reset the global storage service (primarily for testing).

**Signature**: `reset_storage_service() -> None`

## Usage Examples

### Basic File Operations
```python
from app.services.environment_storage import get_storage_service

# Get the storage service (automatically configured for environment)
storage_service = get_storage_service()

# Save an uploaded file
filename, file_path = await storage_service.save_uploaded_file(uploaded_file)

# Generate a URL for accessing the file
file_url = storage_service.get_file_url(file_path)

# Get file information
file_info = storage_service.get_file_info(file_path)

# Delete a file
success = storage_service.delete_file(file_path)
```

### Demo Service Integration
```python
# In demo service - automatically works with both local and S3
storage_service = get_storage_service()
original_image_url = storage_service.get_file_url(drawing.file_path)
saliency_map_url = storage_service.get_file_url(interpretability.saliency_map_path)
```

## Migration from FileStorageService

### Services Updated
- **Demo Service** (`app/services/demo_service.py`) - Now uses environment-aware storage for image URLs
- **Drawings Endpoint** (`app/api/api_v1/endpoints/drawings.py`) - Updated for file serving
- **Additional endpoints** - As part of S3 storage integration

### Benefits of Migration
1. **Environment Consistency**: Same code works in local and production
2. **S3 Integration**: Automatic S3 support in production environments
3. **Presigned URLs**: Secure, time-limited access to S3 files
4. **Unified Interface**: Single service for all storage operations

## Error Handling

### StorageError
Custom exception for storage operations

**Common Scenarios**:
- S3 bucket access denied
- Local directory permissions
- File not found
- Network connectivity issues

### Graceful Fallbacks
- Missing AWS credentials: Falls back to local storage with warnings
- S3 unavailable: Logs errors and attempts local fallback where possible
- Invalid file paths: Returns appropriate error responses

## Performance Considerations

### S3 Backend
- **Presigned URLs**: 1-hour expiration for security
- **API Calls**: Minimized through caching where appropriate
- **Latency**: Initial URL generation may have slight delay

### Local Backend
- **Direct Access**: No additional latency
- **File System**: Standard filesystem performance characteristics

## Security

### S3 Security
- **Presigned URLs**: Time-limited access (1 hour)
- **IAM Permissions**: Requires appropriate S3 access permissions
- **Bucket Policies**: Should restrict public access

### Local Security
- **File Permissions**: Standard filesystem permissions
- **Directory Access**: Restricted to configured directories

## Validation

This contract is automatically validated against the implementation in:
- Source file: `app/services/environment_storage.py`
- Last updated: 2025-01-04

## Dependencies

### Required
- `fastapi` - For UploadFile type
- `aiofiles` - For async file operations
- `pathlib` - For path handling

### Optional
- `boto3` - Required for S3 backend functionality
- AWS credentials and permissions for S3 access