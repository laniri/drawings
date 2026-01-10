# POST /api/v1/config/reset

## Summary
Reset System

## Description
Reset system configuration and models.

WARNING: This endpoint deactivates all models and clears caches.
Use with caution in production environments.

## Tags
configuration

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| confirm | query | boolean | No | No description |

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
POST /api/v1/config/reset
Content-Type: application/json
Accept: application/json
```

