# PUT /api/config/threshold

## Summary
Update Threshold Settings

## Description
Update global threshold settings.

This endpoint recalculates thresholds for all active models
using the specified percentile value from the request body.

## Tags
configuration

## Parameters
No parameters required.

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
{
  "success": true,
  "message": "example_string",
  "data": {}
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
PUT /api/config/threshold
Content-Type: application/json
Accept: application/json
```

