# DELETE /api/v1/drawings/{drawing_id}

## Summary
Delete Drawing

## Description
Delete drawing and associated data.

## Tags
drawings

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| drawing_id | path | integer | Yes | No description |

## Responses

### 204 - Successful Response

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
DELETE /api/v1/drawings/{drawing_id}
Content-Type: application/json
Accept: application/json
```

