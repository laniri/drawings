# POST /api/security/validate/vpc

## Summary
Validate Vpc

## Description
Validate VPC configuration for security compliance.

Checks the specified VPC for proper subnet configuration,
private subnet isolation, and network security best practices.

## Tags
security

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| vpc_id | query | string | Yes | VPC ID to validate |

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
POST /api/security/validate/vpc
Content-Type: application/json
Accept: application/json
```

