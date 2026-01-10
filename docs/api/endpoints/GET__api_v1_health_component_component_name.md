# GET /api/v1/health/component/{component_name}

## Summary
Component-specific health check

## Description
Get health status for a specific component.

## Tags
health

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| component_name | path | string | Yes | No description |

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
GET /api/v1/health/component/{component_name}
Content-Type: application/json
Accept: application/json
```

