# POST /api/v1/interpretability/{analysis_id}/annotate

## Summary
Add Annotation

## Description
Add user annotations to interpretability results.

This endpoint allows users to add their own notes and observations
to interpretability results for future reference.

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
  "region_id": "example_string",
  "annotation_text": "example_string",
  "annotation_type": "example_string",
  "user_id": {}
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
POST /api/v1/interpretability/{analysis_id}/annotate
Content-Type: application/json
Accept: application/json
```

