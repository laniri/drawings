# POST /api/v1/analysis/batch

## Summary
Batch Analyze

## Description
Batch analyze multiple drawings.

This endpoint accepts a list of drawing IDs and processes them
in the background, returning a batch ID for progress tracking.

## Tags
analysis

## Parameters
No parameters required.

## Request Body
Request body required

### Request Body Examples

**application/json**:
```json
{
  "drawing_ids": [
    42,
    42
  ],
  "force_reanalysis": true
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
POST /api/v1/analysis/batch
Content-Type: application/json
Accept: application/json
```

