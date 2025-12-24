# GET /api/v1/documentation/preview/{category}

## Summary
Preview Documentation Changes

## Description
Preview documentation changes before generation.

Shows what would be generated for a specific category or file
without actually writing the files.

## Parameters
- **category** (path): No description
- **file_path** (query): Specific file to preview

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
GET /api/v1/documentation/preview/{category}
```
