# POST /api/v1/documentation/search/index

## Summary
Rebuild Search Index

## Description
Rebuild the search index.

Rebuilds the search index from all documentation files.
Use force=true to completely rebuild the index.

## Tags
documentation

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| force | query | boolean | No | Force complete reindexing |

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
POST /api/v1/documentation/search/index
Content-Type: application/json
Accept: application/json
```

