# POST /api/v1/interpretability/{analysis_id}/export

## Summary
Export Interpretability Results

## Description
Export interpretability results in multiple formats (PDF, PNG, CSV, JSON, HTML).

This endpoint allows users to export comprehensive interpretability reports
with customizable options for different use cases.

## Tags
interpretability

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| analysis_id | path | integer | Yes | No description |

## Request Body
Request body required

### Request Body Examples

**application/json**:
```json
{
  "format": "example_string",
  "include_annotations": true,
  "include_comparisons": true,
  "simplified_version": true,
  "export_options": {}
}
```


## Responses

### 200 - Successful Response

**application/json**:
```json
{
  "export_id": "example_string",
  "file_path": "example_string",
  "file_url": "example_string",
  "format": "example_string",
  "file_size": 42,
  "created_at": "example_string",
  "expires_at": {}
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
POST /api/v1/interpretability/{analysis_id}/export
Content-Type: application/json
Accept: application/json
```

