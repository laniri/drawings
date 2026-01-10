# POST /api/v1/models/train

## Summary
Train Age Group Model

## Description
Train new age group model.

This endpoint starts training a new autoencoder model for the specified
age range. Training is performed in the background and progress can be
tracked using the returned job ID.

## Tags
models

## Parameters
No parameters required.

## Request Body
Request body required

### Request Body Examples

**application/json**:
```json
{
  "age_min": 3.14,
  "age_max": 3.14,
  "model_type": "example_string",
  "vision_model": "example_string",
  "min_samples": 42
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
POST /api/v1/models/train
Content-Type: application/json
Accept: application/json
```

