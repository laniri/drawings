# POST /api/v1/backup/full

## Summary
Create full system backup

## Description
Create a full system backup including database and files.

## Parameters
- **include_files** (query): Include uploaded files and generated content

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
POST /api/v1/backup/full
```
