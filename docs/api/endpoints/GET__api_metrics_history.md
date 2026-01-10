# GET /api/metrics/history

## Summary
Historical metrics

## Description
Get historical system metrics.

## Tags
health

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| hours | query | integer | No | Hours of history to retrieve |

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
GET /api/metrics/history
Content-Type: application/json
Accept: application/json
```

