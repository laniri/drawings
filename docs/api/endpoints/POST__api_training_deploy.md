# POST /api/training/deploy

## Summary
Deploy Trained Model

## Description
Deploy trained model parameters to production system.

This endpoint loads trained model parameters and creates a new
age group model for production use.

## Tags
training

## Parameters
No parameters required.

## Request Body
Request body required

### Request Body Examples

**application/json**:
```json
{
  "model_parameters_path": "example_string",
  "age_group_min": 3.14,
  "age_group_max": 3.14,
  "replace_existing": true
}
```


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
POST /api/training/deploy
Content-Type: application/json
Accept: application/json
```

