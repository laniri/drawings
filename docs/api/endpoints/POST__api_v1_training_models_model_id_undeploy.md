# POST /api/v1/training/models/{model_id}/undeploy

## Summary
Undeploy Model

## Description
Undeploy (deactivate) a deployed model.

This endpoint deactivates a deployed model, removing it from
active use in the production system.

## Parameters
- **model_id** (path): No description

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
POST /api/v1/training/models/{model_id}/undeploy
```
