# PUT /api/v1/models/{model_id}/threshold

## Summary
Update Model Threshold

## Description
Update model threshold.

This endpoint allows updating the anomaly detection threshold
for a specific age group model. The threshold can be set directly
or calculated from a percentile of validation data.

## Tags
models

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| model_id | path | integer | Yes | No description |

## Request Body
Request body required

### Request Body Examples

**application/json**:
```json
{
  "threshold": 3.14,
  "percentile": {}
}
```


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
PUT /api/v1/models/{model_id}/threshold
Content-Type: application/json
Accept: application/json
```

