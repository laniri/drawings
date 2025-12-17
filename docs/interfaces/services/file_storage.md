# File Storage Service

File storage service for handling uploaded drawings and generated files.

This module provides functionality for storing, organizing, and managing
uploaded drawing files and generated analysis results.

## Class: FileStorageError

Custom exception for file storage operations (deprecated, use StorageError)

## Class: FileStorageService

Service for managing file storage operations

### generate_unique_filename

Generate a unique filename while preserving the original extension

Args:
    original_filename: Original filename from upload
    prefix: Optional prefix for the filename
    
Returns:
    Unique filename with timestamp and UUID

**Signature**: `generate_unique_filename(original_filename, prefix)`

### get_file_url

Generate a URL for accessing a stored file

Args:
    file_path: Full file path or relative path from static directory
    base_url: Base URL for static files
    
Returns:
    URL for accessing the file

**Signature**: `get_file_url(file_path, base_url)`

### delete_file

Delete a file from the storage system

Args:
    file_path: Path to the file to delete
    
Returns:
    True if deletion was successful, False otherwise

**Signature**: `delete_file(file_path)`

### get_file_info

Get information about a stored file

Args:
    file_path: Path to the file
    
Returns:
    Dictionary with file information or None if file doesn't exist

**Signature**: `get_file_info(file_path)`

### cleanup_old_files

Clean up old files from a directory

Args:
    directory: Directory to clean up
    max_age_days: Maximum age of files to keep in days
    
Returns:
    Number of files deleted

**Signature**: `cleanup_old_files(directory, max_age_days)`

### get_storage_stats

Get storage statistics

Returns:
    Dictionary with storage statistics

**Signature**: `get_storage_stats()`

