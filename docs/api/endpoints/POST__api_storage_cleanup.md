# POST /api/storage/cleanup

## Summary
Clean up temporary and orphaned files

## Description
Clean up temporary files and orphaned data.

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
POST /api/storage/cleanup
Content-Type: application/json
Accept: application/json
```

