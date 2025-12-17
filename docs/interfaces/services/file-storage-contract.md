# File Storage Contract

## Overview
Service contract for File Storage (service)

**Source File**: `app/services/file_storage.py`

## Interface Specification

### Classes

#### FileStorageError

Custom exception for file storage operations (deprecated, use StorageError)

**Inherits from**: StorageError

#### FileStorageService

Service for managing file storage operations

## Methods

### generate_unique_filename

Generate a unique filename while preserving the original extension

Args:
    original_filename: Original filename from upload
    prefix: Optional prefix for the filename
    
Returns:
    Unique filename with timestamp and UUID

**Signature**: `generate_unique_filename(original_filename: str, prefix: str) -> str`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `original_filename` | `str` | Parameter description |
| `prefix` | `str` | Parameter description |

**Returns**: `str`

### get_file_url

Generate a URL for accessing a stored file

Args:
    file_path: Full file path or relative path from static directory
    base_url: Base URL for static files
    
Returns:
    URL for accessing the file

**Signature**: `get_file_url(file_path: str, base_url: str) -> str`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `file_path` | `str` | Parameter description |
| `base_url` | `str` | Parameter description |

**Returns**: `str`

### delete_file

Delete a file from the storage system

Args:
    file_path: Path to the file to delete
    
Returns:
    True if deletion was successful, False otherwise

**Signature**: `delete_file(file_path: str) -> bool`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `file_path` | `str` | Parameter description |

**Returns**: `bool`

### get_file_info

Get information about a stored file

Args:
    file_path: Path to the file
    
Returns:
    Dictionary with file information or None if file doesn't exist

**Signature**: `get_file_info(file_path: str) -> <ast.Subscript object at 0x110436b10>`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `file_path` | `str` | Parameter description |

**Returns**: `<ast.Subscript object at 0x110436b10>`

### cleanup_old_files

Clean up old files from a directory

Args:
    directory: Directory to clean up
    max_age_days: Maximum age of files to keep in days
    
Returns:
    Number of files deleted

**Signature**: `cleanup_old_files(directory: str, max_age_days: int) -> int`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `directory` | `str` | Parameter description |
| `max_age_days` | `int` | Parameter description |

**Returns**: `int`

### get_storage_stats

Get storage statistics

Returns:
    Dictionary with storage statistics

**Signature**: `get_storage_stats() -> Dict[str, Any]`

**Returns**: `Dict[str, Any]`

## Defined Interfaces

### FileStorageServiceInterface

**Type**: Protocol
**Implemented by**: FileStorageService

**Methods**:

- `generate_unique_filename(original_filename: str, prefix: str) -> str`
- `get_file_url(file_path: str, base_url: str) -> str`
- `delete_file(file_path: str) -> bool`
- `get_file_info(file_path: str) -> <ast.Subscript object at 0x110436b10>`
- `cleanup_old_files(directory: str, max_age_days: int) -> int`
- `get_storage_stats() -> Dict[str, Any]`

## Usage Examples

```python
# Example usage of the service contract
# TODO: Add specific usage examples
```

## Validation

This contract is automatically validated against the implementation in:
- Source file: `app/services/file_storage.py`
- Last validated: 2025-12-16 15:47:04

