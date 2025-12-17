# POST /api/v1/drawings/batch/upload

## Summary
Batch Upload Drawings

## Description
Upload multiple drawings in batch.

This endpoint accepts multiple files and processes them in the background.
Returns an upload ID for tracking progress.

## Parameters
No parameters

## Request Body
Request body required

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
POST /api/v1/drawings/batch/upload
```
