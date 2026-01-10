# DELETE /api/v1/models/{model_id}

## Summary
Delete Model

## Description
Delete (deactivate) an age group model.

This endpoint deactivates a model rather than permanently deleting it
to preserve analysis history.

## Tags
models

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| model_id | path | integer | Yes | No description |

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
DELETE /api/v1/models/{model_id}
Content-Type: application/json
Accept: application/json
```

