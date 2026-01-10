# POST /api/v1/security/audit/comprehensive

## Summary
Comprehensive Security Audit

## Description
Perform comprehensive security audit of AWS resources.

Validates all specified AWS resources for security compliance,
including IAM roles, S3 buckets, security groups, VPC configuration,
and encryption settings.

## Tags
security

## Parameters
No parameters required.

## Request Body
Request body required

### Request Body Examples

**application/json**:
```json
{
  "iam_role_arn": {},
  "s3_buckets": {},
  "security_group_ids": {},
  "vpc_id": {}
}
```


## Responses

### 200 - Successful Response

**application/json**:
```json
{
  "overall_compliant": true,
  "total_violations": 42,
  "total_warnings": 42,
  "components": {},
  "summary": {}
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
POST /api/v1/security/audit/comprehensive
Content-Type: application/json
Accept: application/json
```

