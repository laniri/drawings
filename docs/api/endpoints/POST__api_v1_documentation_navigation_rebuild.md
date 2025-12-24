# POST /api/v1/documentation/navigation/rebuild

## Summary
Rebuild Navigation Structure

## Description
Rebuild navigation structure.

Rebuilds the navigation structure and cross-reference index
from all documentation files.

## Parameters
- **force** (query): Force complete rebuild

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
POST /api/v1/documentation/navigation/rebuild
```
