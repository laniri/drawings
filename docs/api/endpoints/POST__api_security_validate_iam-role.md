# POST /api/security/validate/iam-role

## Summary
Validate Iam Role

## Description
Validate IAM role for least-privilege compliance.

Checks the specified IAM role for overly broad permissions,
dangerous policy attachments, and compliance with security best practices.

## Tags
security

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| role_arn | query | string | Yes | IAM role ARN to validate |

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
POST /api/security/validate/iam-role
Content-Type: application/json
Accept: application/json
```

