# GET /api/documentation/navigation/{document_id}

## Summary
Get Navigation Context

## Description
Get navigation context for a document.

Returns comprehensive navigation context including breadcrumbs,
cross-references, related content, and sequential navigation.

## Tags
documentation

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| document_id | path | string | Yes | No description |

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
GET /api/documentation/navigation/{document_id}
Content-Type: application/json
Accept: application/json
```

