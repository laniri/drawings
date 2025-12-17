# Error Specifications

This document provides comprehensive information about error responses from the API.

## Error Response Format

All API errors follow a consistent format to ensure predictable error handling:

```json
{
  "detail": "Human-readable error message",
  "error_code": "MACHINE_READABLE_CODE",
  "timestamp": "2023-12-01T10:30:00Z",
  "path": "/api/v1/endpoint",
  "request_id": "req_123456789"
}
```

## HTTP Status Codes

### 2xx Success
- **200 OK**: Request successful
- **201 Created**: Resource created successfully
- **202 Accepted**: Request accepted for processing
- **204 No Content**: Request successful, no content returned

### 4xx Client Errors
- **400 Bad Request**: Invalid request syntax or parameters
- **401 Unauthorized**: Authentication required or invalid
- **403 Forbidden**: Access denied (valid auth but insufficient permissions)
- **404 Not Found**: Requested resource not found
- **405 Method Not Allowed**: HTTP method not supported for endpoint
- **409 Conflict**: Request conflicts with current resource state
- **422 Unprocessable Entity**: Request syntax valid but semantically incorrect
- **429 Too Many Requests**: Rate limit exceeded

### 5xx Server Errors
- **500 Internal Server Error**: Unexpected server error
- **502 Bad Gateway**: Invalid response from upstream server
- **503 Service Unavailable**: Server temporarily unavailable
- **504 Gateway Timeout**: Upstream server timeout

## Common Error Scenarios

### Validation Errors (422)

When request data fails validation:

```json
{
  "detail": [
    {
      "loc": ["body", "age"],
      "msg": "ensure this value is greater than 0",
      "type": "value_error.number.not_gt",
      "ctx": {"limit_value": 0}
    },
    {
      "loc": ["body", "file"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**Fields**:
- `loc`: Location of the error (path to the field)
- `msg`: Human-readable error message
- `type`: Machine-readable error type
- `ctx`: Additional context (when available)

### File Upload Errors

#### Invalid File Type (400)
```json
{
  "detail": "Invalid file type. Supported formats: PNG, JPEG, BMP",
  "error_code": "INVALID_FILE_TYPE",
  "supported_formats": ["image/png", "image/jpeg", "image/bmp"]
}
```

#### File Too Large (413)
```json
{
  "detail": "File size exceeds maximum limit of 10MB",
  "error_code": "FILE_TOO_LARGE",
  "max_size_bytes": 10485760,
  "received_size_bytes": 15728640
}
```

### Authentication Errors

#### Missing Authentication (401)
```json
{
  "detail": "Authentication credentials were not provided",
  "error_code": "AUTHENTICATION_REQUIRED"
}
```

#### Invalid Token (401)
```json
{
  "detail": "Invalid or expired authentication token",
  "error_code": "INVALID_TOKEN"
}
```

### Resource Errors

#### Not Found (404)
```json
{
  "detail": "Drawing with ID 'abc123' not found",
  "error_code": "RESOURCE_NOT_FOUND",
  "resource_type": "drawing",
  "resource_id": "abc123"
}
```

#### Conflict (409)
```json
{
  "detail": "Drawing with this filename already exists",
  "error_code": "RESOURCE_CONFLICT",
  "conflicting_field": "filename"
}
```

### Rate Limiting (429)

```json
{
  "detail": "Rate limit exceeded. Try again in 60 seconds",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "retry_after": 60,
  "limit": 100,
  "window": "1h"
}
```

### Server Errors (5xx)

#### Internal Server Error (500)
```json
{
  "detail": "An unexpected error occurred. Please try again later",
  "error_code": "INTERNAL_SERVER_ERROR",
  "request_id": "req_123456789"
}
```

#### Service Unavailable (503)
```json
{
  "detail": "ML model service temporarily unavailable",
  "error_code": "SERVICE_UNAVAILABLE",
  "service": "ml_inference",
  "retry_after": 30
}
```

## Error Handling Best Practices

### Client Implementation

1. **Always check HTTP status codes** before processing response body
2. **Parse error details** from the response body for user-friendly messages
3. **Implement retry logic** for 5xx errors and 429 rate limiting
4. **Log error details** including request_id for debugging

### Example Error Handling (Python)

```python
import requests
import time

def api_request_with_retry(url, **kwargs):
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            response = requests.request(**kwargs, url=url)
            
            if response.status_code < 400:
                return response.json()
            
            elif response.status_code == 429:
                # Rate limited - check retry-after header
                retry_after = int(response.headers.get('Retry-After', retry_delay))
                time.sleep(retry_after)
                continue
            
            elif 500 <= response.status_code < 600:
                # Server error - retry with exponential backoff
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                    continue
            
            # Client error or final attempt - raise exception
            error_data = response.json()
            raise APIError(
                status_code=response.status_code,
                detail=error_data.get('detail', 'Unknown error'),
                error_code=error_data.get('error_code')
            )
            
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
                continue
            raise

class APIError(Exception):
    def __init__(self, status_code, detail, error_code=None):
        self.status_code = status_code
        self.detail = detail
        self.error_code = error_code
        super().__init__(f"{status_code}: {detail}")
```

## Error Code Reference

| Error Code | HTTP Status | Description |
|------------|-------------|-------------|
| `VALIDATION_ERROR` | 422 | Request data validation failed |
| `INVALID_FILE_TYPE` | 400 | Unsupported file format |
| `FILE_TOO_LARGE` | 413 | File exceeds size limit |
| `AUTHENTICATION_REQUIRED` | 401 | Missing authentication |
| `INVALID_TOKEN` | 401 | Invalid or expired token |
| `INSUFFICIENT_PERMISSIONS` | 403 | Access denied |
| `RESOURCE_NOT_FOUND` | 404 | Requested resource not found |
| `RESOURCE_CONFLICT` | 409 | Resource already exists |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_SERVER_ERROR` | 500 | Unexpected server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily down |
| `ML_MODEL_ERROR` | 500 | ML inference failed |
| `DATABASE_ERROR` | 500 | Database operation failed |

---

*For additional support, include the `request_id` from error responses when reporting issues.*

## Error Schema Definitions

### HTTPValidationError

```json
{
  "properties": {
    "detail": {
      "items": {
        "$ref": "#/components/schemas/ValidationError"
      },
      "type": "array",
      "title": "Detail"
    }
  },
  "type": "object",
  "title": "HTTPValidationError"
}
```

### ValidationError

```json
{
  "properties": {
    "loc": {
      "items": {
        "anyOf": [
          {
            "type": "string"
          },
          {
            "type": "integer"
          }
        ]
      },
      "type": "array",
      "title": "Location"
    },
    "msg": {
      "type": "string",
      "title": "Message"
    },
    "type": {
      "type": "string",
      "title": "Error Type"
    }
  },
  "type": "object",
  "required": [
    "loc",
    "msg",
    "type"
  ],
  "title": "ValidationError"
}
```

