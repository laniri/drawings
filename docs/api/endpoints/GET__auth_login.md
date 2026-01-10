# GET /auth/login

## Summary
Login Page

## Description
Display login page.

Args:
    request: FastAPI request object
    redirect: URL to redirect to after successful login

Returns:
    HTML login page

## Tags
authentication

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| redirect | query | unknown | No | No description |

## Responses

### 200 - Successful Response

**text/html**:
```json
"example_string"
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
GET /auth/login
Content-Type: application/json
Accept: application/json
```

