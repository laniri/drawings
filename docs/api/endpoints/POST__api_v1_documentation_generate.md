# POST /api/v1/documentation/generate

## Summary
Generate Documentation

## Description
Trigger documentation generation.

Starts documentation generation process in the background.
Use the status endpoint to monitor progress.

## Parameters
No parameters

## Request Body
Request body required

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
POST /api/v1/documentation/generate
```
