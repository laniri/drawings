# POST /api/v1/training/models/validate

## Summary
Validate Exported Model

## Description
Validate exported model for compatibility and integrity.

This endpoint performs comprehensive validation of an exported model,
checking file integrity, compatibility, and performance metrics.

## Tags
training

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| model_id | query | string | Yes | No description |

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
POST /api/v1/training/models/validate
Content-Type: application/json
Accept: application/json
```

