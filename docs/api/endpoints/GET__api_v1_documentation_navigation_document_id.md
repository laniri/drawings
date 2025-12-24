# GET /api/v1/documentation/navigation/{document_id}

## Summary
Get Navigation Context

## Description
Get navigation context for a document.

Returns comprehensive navigation context including breadcrumbs,
cross-references, related content, and sequential navigation.

## Parameters
- **document_id** (path): No description

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
GET /api/v1/documentation/navigation/{document_id}
```
