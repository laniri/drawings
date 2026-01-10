# POST /api/alerts/thresholds

## Summary
Update alert thresholds

## Description
Update system alert thresholds.

## Tags
health

## Parameters
No parameters required.

## Request Body
Request body required

### Request Body Examples

**application/json**:
```json
{}
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
POST /api/alerts/thresholds
Content-Type: application/json
Accept: application/json
```

