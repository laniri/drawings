# GET /api/security/status

## Summary
Get Security Status

## Description
Get current security service status and configuration.

Returns information about security service initialization,
AWS client availability, and current security policy.

## Tags
security

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
GET /api/security/status
Content-Type: application/json
Accept: application/json
```

