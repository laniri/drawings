# POST /api/v1/training/models/validate

## Summary
Validate Exported Model

## Description
Validate exported model for compatibility and integrity.

This endpoint performs comprehensive validation of an exported model,
checking file integrity, compatibility, and performance metrics.

## Parameters
- **model_id** (query): No description

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
POST /api/v1/training/models/validate
```
