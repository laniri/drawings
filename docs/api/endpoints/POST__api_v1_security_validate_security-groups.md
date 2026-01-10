# POST /api/v1/security/validate/security-groups

## Summary
Validate Security Groups

## Description
Validate security group configurations for minimal exposure.

Checks the specified security groups for overly permissive rules,
open ports, and compliance with network security best practices.

## Tags
security

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| security_group_ids | query | array | Yes | Security group IDs to validate |

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
POST /api/v1/security/validate/security-groups
Content-Type: application/json
Accept: application/json
```

