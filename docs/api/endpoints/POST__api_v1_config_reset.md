# POST /api/v1/config/reset

## Summary
Reset System

## Description
Reset system configuration and models.

WARNING: This endpoint deactivates all models and clears caches.
Use with caution in production environments.

## Parameters
- **confirm** (query): No description

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
POST /api/v1/config/reset
```
