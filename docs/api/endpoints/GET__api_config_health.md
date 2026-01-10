# GET /api/config/health

## Summary
Health Check

## Description
System health check endpoint.

This endpoint provides information about the health and status
of various system components.

## Tags
configuration

## Parameters
No parameters required.

## Responses

### 200 - Successful Response

**application/json**:
```json
{
  "status": "example_string",
  "timestamp": "example_string",
  "version": "example_string",
  "database": "example_string",
  "models": "example_string",
  "storage": "example_string"
}
```


## Complete Request Example

```http
GET /api/config/health
Content-Type: application/json
Accept: application/json
```

