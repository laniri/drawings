# GET /monitoring/alerts

## Summary
Get Recent Alerts

## Description
Get recent alerts for monitoring.

## Tags
untagged

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| limit | query | integer | No | No description |

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
GET /monitoring/alerts
Content-Type: application/json
Accept: application/json
```

