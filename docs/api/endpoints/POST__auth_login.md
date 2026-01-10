# POST /auth/login

## Summary
Login

## Description
Process login form submission.

Args:
    request: FastAPI request object
    response: FastAPI response object
    password: Admin password
    redirect_url: URL to redirect to after successful login

Returns:
    Redirect response or error page

## Tags
authentication

## Parameters
No parameters required.

## Request Body
Request body required

### Request Body Examples

**application/x-www-form-urlencoded**:
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
POST /auth/login
Content-Type: application/json
Accept: application/json
```

