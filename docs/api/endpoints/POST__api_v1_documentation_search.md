# POST /api/v1/documentation/search

## Summary
Search Documentation

## Description
Search documentation with advanced filtering and faceting.

Provides full-text search across all documentation with relevance scoring,
faceted filtering, and intelligent suggestions.

## Tags
documentation

## Parameters
No parameters required.

## Request Body
Request body required

### Request Body Examples

**application/json**:
```json
{
  "query": "example_string",
  "doc_types": {},
  "tags": {},
  "limit": 42,
  "offset": 42,
  "include_content": true,
  "highlight": true
}
```


## Responses

### 200 - Successful Response

**application/json**:
```json
{
  "results": [
    {
      "id": "example_string",
      "title": "example_string",
      "doc_type": "example_string",
      "url": "example_string",
      "score": 3.14,
      "snippet": {},
      "highlights": [
        "example_string",
        "example_string"
      ],
      "tags": [
        "example_string",
        "example_string"
      ],
      "last_modified": "example_string"
    },
    {
      "id": "example_string",
      "title": "example_string",
      "doc_type": "example_string",
      "url": "example_string",
      "score": 3.14,
      "snippet": {},
      "highlights": [
        "example_string",
        "example_string"
      ],
      "tags": [
        "example_string",
        "example_string"
      ],
      "last_modified": "example_string"
    }
  ],
  "total_count": 42,
  "query_time": 3.14,
  "facets": {},
  "suggestions": [
    "example_string",
    "example_string"
  ],
  "query": "example_string"
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
POST /api/v1/documentation/search
Content-Type: application/json
Accept: application/json
```

