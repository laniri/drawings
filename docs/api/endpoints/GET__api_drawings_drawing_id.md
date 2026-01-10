# GET /api/drawings/{drawing_id}

## Summary
Get Drawing

## Description
Retrieve drawing details by ID.

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
GET /api/drawings/{drawing_id}
Content-Type: application/json
Accept: application/json
```

