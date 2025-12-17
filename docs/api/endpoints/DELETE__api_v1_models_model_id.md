# DELETE /api/v1/models/{model_id}

## Summary
Delete Model

## Description
Delete (deactivate) an age group model.

This endpoint deactivates a model rather than permanently deleting it
to preserve analysis history.

## Parameters
- **model_id** (path): No description

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
DELETE /api/v1/models/{model_id}
```
