# GET /api/interpretability/examples

## Summary
Get Example Patterns

## Description
Get example interpretation patterns for educational purposes.

This endpoint provides a gallery of common interpretation patterns
with explanations suitable for different user roles.

## Tags
interpretability

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| age_group | query | unknown | No | No description |
| user_role | query | string | No | No description |

## Responses

### 200 - Successful Response

**application/json**:
```json
[
  {},
  {}
]
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
GET /api/interpretability/examples
Content-Type: application/json
Accept: application/json
```

