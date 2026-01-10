# GET /auth/status

## Summary
Session Status

## Description
Get current session status.

Args:
    request: FastAPI request object

Returns:
    Session status information

## Tags
authentication

## Parameters
No parameters required.

## Responses

### 200 - Successful Response

**application/json**:
```json
{
  "authenticated": true,
  "session_info": {},
  "expires_in": {}
}
```


## Complete Request Example

```http
GET /auth/status
Content-Type: application/json
Accept: application/json
```

