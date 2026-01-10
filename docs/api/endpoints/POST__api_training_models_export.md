# POST /api/training/models/export

## Summary
Export Model From Training Job

## Description
Export trained model from training job in production-compatible format.

This endpoint exports a trained model from a completed training job,
creating a production-ready model file with metadata and validation.

## Tags
training

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| training_job_id | query | integer | Yes | No description |
| age_group_min | query | number | Yes | No description |
| age_group_max | query | number | Yes | No description |
| export_format | query | string | No | No description |

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
POST /api/training/models/export
Content-Type: application/json
Accept: application/json
```

