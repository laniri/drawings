# GET /api/drawings/{drawing_id}/file

## Summary
Get Drawing File

## Description
Retrieve the actual drawing file.

## Tags
drawings

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| drawing_id | path | integer | Yes | No description |

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
GET /api/drawings/{drawing_id}/file
Content-Type: application/json
Accept: application/json
```

