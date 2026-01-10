# POST /api/drawings/upload

## Summary
Upload Drawing

## Description
Upload drawing with metadata.

This endpoint accepts multipart form data with an image file and metadata.
The image is validated, preprocessed, and stored along with the metadata.

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
  "file": "example_string",
  "age_years": 3.14,
  "subject": {},
  "expert_label": {},
  "drawing_tool": {},
  "prompt": {}
}
```


## Responses

### 201 - Successful Response

**application/json**:
```json
{
  "id": 42,
  "filename": "example_string",
  "age_years": 3.14,
  "subject": {},
  "expert_label": {},
  "drawing_tool": {},
  "prompt": {},
  "upload_timestamp": "example_string"
}
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
POST /api/drawings/upload
Content-Type: application/json
Accept: application/json
```

