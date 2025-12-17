# POST /api/v1/training/models/deploy

## Summary
Deploy Exported Model

## Description
Deploy exported model to production environment.

This endpoint deploys an exported model to the production system,
making it available for anomaly detection in the specified age group.

## Parameters
- **model_export_path** (query): No description
- **age_group_min** (query): No description
- **age_group_max** (query): No description
- **replace_existing** (query): No description
- **validate_before_deployment** (query): No description
- **backup_existing** (query): No description

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
POST /api/v1/training/models/deploy
```
