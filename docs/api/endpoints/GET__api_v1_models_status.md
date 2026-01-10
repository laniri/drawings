# GET /api/v1/models/status

## Summary
Get Model Status

## Description
Get model training and system status.

This endpoint provides an overview of the model management system,
including counts of models in different states and overall system health.

## Tags
models

## Parameters
No parameters required.

## Responses

### 200 - Successful Response

**application/json**:
```json
{
  "total_models": 42,
  "active_models": 42,
  "training_models": 42,
  "failed_models": 42,
  "total_drawings": 42,
  "total_analyses": 42,
  "system_status": "example_string",
  "last_training": {}
}
```


## Complete Request Example

```http
GET /api/v1/models/status
Content-Type: application/json
Accept: application/json
```

