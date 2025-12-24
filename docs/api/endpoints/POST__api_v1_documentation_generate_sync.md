# POST /api/v1/documentation/generate/sync

## Summary
Generate Documentation Sync

## Description
Generate documentation synchronously.

Runs documentation generation and waits for completion.
Use this for smaller generation tasks or when immediate results are needed.

## Parameters
No parameters

## Request Body
Request body required

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
POST /api/v1/documentation/generate/sync
```
