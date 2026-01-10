# POST /api/v1/database/schedule-backups

## Summary
Schedule Automated Backups

## Description
Schedule automated database backups.

- **interval_hours**: Backup interval in hours (default: 6)

## Tags
database

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| interval_hours | query | integer | No | No description |

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
POST /api/v1/database/schedule-backups
Content-Type: application/json
Accept: application/json
```

