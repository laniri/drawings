# POST /api/security/validate/s3-bucket

## Summary
Validate S3 Bucket

## Description
Validate S3 bucket encryption and security configuration.

Checks the specified S3 bucket for proper encryption configuration,
public access blocks, and security compliance.

## Tags
security

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| bucket_name | query | string | Yes | S3 bucket name to validate |

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
POST /api/security/validate/s3-bucket
Content-Type: application/json
Accept: application/json
```

