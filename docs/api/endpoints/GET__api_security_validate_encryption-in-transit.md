# GET /api/security/validate/encryption-in-transit

## Summary
Validate Encryption In Transit

## Description
Validate encryption in transit configuration.

Checks the current deployment for proper HTTPS enforcement,
TLS configuration, and secure communication protocols.

## Tags
security

## Parameters
No parameters required.

## Responses

### 200 - Successful Response

**application/json**:
```json
{
  "is_compliant": true,
  "violations": [
    "example_string",
    "example_string"
  ],
  "warnings": [
    "example_string",
    "example_string"
  ],
  "recommendations": [
    "example_string",
    "example_string"
  ],
  "details": {}
}
```


## Complete Request Example

```http
GET /api/security/validate/encryption-in-transit
Content-Type: application/json
Accept: application/json
```

