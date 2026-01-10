# POST /api/drawings/batch/upload

## Summary
Batch Upload Drawings

## Description
Upload multiple drawings in batch.

This endpoint accepts multiple files and processes them in the background.
Returns an upload ID for tracking progress.

## Tags
drawings

## Parameters
No parameters required.

## Request Body
Request body required

### Request Body Examples

**multipart/form-data**:
```json
{
  "files": [
    "example_string",
    "example_string"
  ]
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
POST /api/drawings/batch/upload
Content-Type: application/json
Accept: application/json
```

