# POST /auth/api/login

## Summary
Api Login

## Description
API endpoint for programmatic login.

Args:
    request: FastAPI request object
    login_data: Login request data

Returns:
    Login response with session token

## Tags
authentication

## Parameters
No parameters required.

## Request Body
Request body required

### Request Body Examples

**application/json**:
```json
{
  "password": "example_string",
  "redirect_url": {}
}
```


## Responses

### 200 - Successful Response

**application/json**:
```json
{
  "success": true,
  "message": "example_string",
  "session_token": {},
  "redirect_url": {}
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
POST /auth/api/login
Content-Type: application/json
Accept: application/json
```

