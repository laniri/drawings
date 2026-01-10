# GET /api/files/markdown

## Summary
Serve Markdown File

## Description
Serve a markdown file from the local filesystem.

Args:
    path: Relative path to markdown file (e.g., "tmp_files/analysis.md")

Returns:
    Markdown file content as plain text

## Tags
files

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| path | query | string | Yes | No description |

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
GET /api/files/markdown
Content-Type: application/json
Accept: application/json
```

