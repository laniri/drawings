# GET /api/v1/security/compliance/report

## Summary
Get Compliance Report

## Description
Generate security compliance report.

Creates a detailed compliance report based on current security
configuration and validation results.

## Tags
security

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| format | query | string | No | Report format: json, csv, or html |

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
GET /api/v1/security/compliance/report
Content-Type: application/json
Accept: application/json
```

