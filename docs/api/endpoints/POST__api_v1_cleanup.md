# POST /api/v1/cleanup

## Summary
Clean up old backups

## Description
Clean up old backup files based on retention policy.

## Tags
backup

## Parameters
No parameters required.

## Responses

### 200 - Successful Response

**application/json**:
```json
{}
```


## Complete Request Example

```http
POST /api/v1/cleanup
Content-Type: application/json
Accept: application/json
```

