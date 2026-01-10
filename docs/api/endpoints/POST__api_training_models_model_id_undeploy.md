# POST /api/training/models/{model_id}/undeploy

## Summary
Undeploy Model

## Description
Undeploy (deactivate) a deployed model.

This endpoint deactivates a deployed model, removing it from
active use in the production system.

## Tags
training

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
POST /api/training/models/{model_id}/undeploy
Content-Type: application/json
Accept: application/json
```

