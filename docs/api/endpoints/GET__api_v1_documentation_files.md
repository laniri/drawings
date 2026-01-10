# GET /api/v1/documentation/files

## Summary
Get Documentation Files

## Description
Get list of documentation files with metadata.

Returns comprehensive list of documentation files with metadata,
filtering, and search capabilities.

## Tags
documentation

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| category | query | unknown | No | Filter by category |
| search | query | unknown | No | Search in file names and content |

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
GET /api/v1/documentation/files
Content-Type: application/json
Accept: application/json
```

