# GET /api/v1/models/age-groups

## Summary
List Age Group Models

## Description
List available age group models.

This endpoint returns all age group models with their status,
sample counts, and threshold information.

## Tags
models

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| active_only | query | boolean | No | No description |

## Responses

### 200 - Successful Response

**application/json**:
```json
{
  "models": [
    {
      "id": 42,
      "age_min": 3.14,
      "age_max": 3.14,
      "model_type": "example_string",
      "vision_model": "example_string",
      "sample_count": 42,
      "threshold": 3.14,
      "status": "example_string",
      "created_timestamp": "example_string",
      "is_active": true
    },
    {
      "id": 42,
      "age_min": 3.14,
      "age_max": 3.14,
      "model_type": "example_string",
      "vision_model": "example_string",
      "sample_count": 42,
      "threshold": 3.14,
      "status": "example_string",
      "created_timestamp": "example_string",
      "is_active": true
    }
  ],
  "total_count": 42,
  "active_count": 42,
  "training_count": 42
}
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
GET /api/v1/models/age-groups
Content-Type: application/json
Accept: application/json
```

