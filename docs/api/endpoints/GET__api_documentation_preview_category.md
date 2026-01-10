# GET /api/documentation/preview/{category}

## Summary
Preview Documentation Changes

## Description
Preview documentation changes before generation.

Shows what would be generated for a specific category or file
without actually writing the files.

## Tags
documentation

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| category | path | string | Yes | No description |
| file_path | query | unknown | No | Specific file to preview |

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
GET /api/documentation/preview/{category}
Content-Type: application/json
Accept: application/json
```

