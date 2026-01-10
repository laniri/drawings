# GET /api/v1/drawings/

## Summary
List Drawings

## Description
List drawings with optional filtering and pagination.

## Tags
drawings

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| age_min | query | unknown | No | No description |
| age_max | query | unknown | No | No description |
| subject | query | unknown | No | No description |
| expert_label | query | unknown | No | No description |
| page | query | integer | No | No description |
| page_size | query | integer | No | No description |

## Responses

### 200 - Successful Response

**application/json**:
```json
{
  "drawings": [
    {
      "id": 42,
      "filename": "example_string",
      "age_years": 3.14,
      "subject": {},
      "expert_label": {},
      "drawing_tool": {},
      "prompt": {},
      "upload_timestamp": "example_string"
    },
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
  ],
  "total_count": 42,
  "page": 42,
  "page_size": 42,
  "total_pages": 42
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
GET /api/v1/drawings/
Content-Type: application/json
Accept: application/json
```

