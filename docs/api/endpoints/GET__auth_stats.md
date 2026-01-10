# GET /auth/stats

## Summary
Auth Stats

## Description
Get authentication service statistics (admin only).

Args:
    request: FastAPI request object

Returns:
    Authentication statistics

## Tags
authentication

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
GET /auth/stats
Content-Type: application/json
Accept: application/json
```

