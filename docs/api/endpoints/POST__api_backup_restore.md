# POST /api/backup/restore

## Summary
Restore from backup

## Description
Restore system from a backup file.

## Tags
backup

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| backup_name | query | string | Yes | No description |

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
POST /api/backup/restore
Content-Type: application/json
Accept: application/json
```

