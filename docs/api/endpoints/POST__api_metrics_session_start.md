# POST /api/metrics/session/start

## Summary
Start User Session

## Description
Manually start a user session (alternative to automatic detection).

Request body should contain:
- ip_address: Client IP address
- user_agent: User agent string

## Tags
metrics

## Parameters
No parameters required.

## Request Body
Request body required

### Request Body Examples

**application/json**:
```json
{}
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
POST /api/metrics/session/start
Content-Type: application/json
Accept: application/json
```

