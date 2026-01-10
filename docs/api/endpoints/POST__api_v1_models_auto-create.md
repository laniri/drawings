# POST /api/v1/models/auto-create

## Summary
Auto Create Age Groups

## Description
Automatically create age group models based on data distribution.

This endpoint analyzes the available drawing data and creates
appropriate age group models with sufficient sample sizes.

## Tags
models

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| force_recreate | query | boolean | No | No description |

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
POST /api/v1/models/auto-create
Content-Type: application/json
Accept: application/json
```

