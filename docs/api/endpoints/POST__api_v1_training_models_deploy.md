# POST /api/v1/training/models/deploy

## Summary
Deploy Exported Model

## Description
Deploy exported model to production environment.

This endpoint deploys an exported model to the production system,
making it available for anomaly detection in the specified age group.

## Tags
training

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| model_export_path | query | string | Yes | No description |
| age_group_min | query | number | Yes | No description |
| age_group_max | query | number | Yes | No description |
| replace_existing | query | boolean | No | No description |
| validate_before_deployment | query | boolean | No | No description |
| backup_existing | query | boolean | No | No description |

## Responses

### 200 - Successful Response

**application/json**:
```json
{}
```

### 422 - Validation Error

**application/json**:
```json
{
  "detail": [
    {
      "loc": [
        {},
        {}
      ],
      "msg": "example_string",
      "type": "example_string"
    },
    {
      "loc": [
        {},
        {}
      ],
      "msg": "example_string",
      "type": "example_string"
    }
  ]
}
```


## Complete Request Example

```http
POST /api/v1/training/models/deploy
Content-Type: application/json
Accept: application/json
```

