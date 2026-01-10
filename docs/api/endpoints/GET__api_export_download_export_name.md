# GET /api/export/download/{export_name}

## Summary
Download exported data

## Description
Download an exported data file.

## Tags
backup

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| export_name | path | string | Yes | No description |

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
GET /api/export/download/{export_name}
Content-Type: application/json
Accept: application/json
```

