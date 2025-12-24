# POST /api/v1/documentation/search/index

## Summary
Rebuild Search Index

## Description
Rebuild the search index.

Rebuilds the search index from all documentation files.
Use force=true to completely rebuild the index.

## Parameters
- **force** (query): Force complete reindexing

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
POST /api/v1/documentation/search/index
```
