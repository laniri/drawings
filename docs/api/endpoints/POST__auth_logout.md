# POST /auth/logout

## Summary
Logout

## Description
Logout user and invalidate session.

Args:
    request: FastAPI request object
    response: FastAPI response object

Returns:
    Redirect to home page

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
POST /auth/logout
Content-Type: application/json
Accept: application/json
```

