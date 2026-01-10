# GET /api/v1/backup/download/{backup_name}

## Summary
Download backup file

## Description
Download a specific backup file.

## Tags
backup

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| backup_name | path | string | Yes | No description |

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
GET /api/v1/backup/download/{backup_name}
Content-Type: application/json
Accept: application/json
```

