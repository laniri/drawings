# POST /api/v1/documentation/navigation/rebuild

## Summary
Rebuild Navigation Structure

## Description
Rebuild navigation structure.

Rebuilds the navigation structure and cross-reference index
from all documentation files.

## Tags
documentation

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| force | query | boolean | No | Force complete rebuild |

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
POST /api/v1/documentation/navigation/rebuild
Content-Type: application/json
Accept: application/json
```

