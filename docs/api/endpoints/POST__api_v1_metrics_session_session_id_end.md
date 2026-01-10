# POST /api/v1/metrics/session/{session_id}/end

## Summary
End User Session

## Description
Manually end a user session.

## Tags
metrics

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| session_id | path | string | Yes | No description |

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
POST /api/v1/metrics/session/{session_id}/end
Content-Type: application/json
Accept: application/json
```

