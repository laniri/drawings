# GET /api/v1/documentation/files

## Summary
Get Documentation Files

## Description
Get list of documentation files with metadata.

Returns comprehensive list of documentation files with metadata,
filtering, and search capabilities.

## Parameters
- **category** (query): Filter by category
- **search** (query): Search in file names and content

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
GET /api/v1/documentation/files
```
