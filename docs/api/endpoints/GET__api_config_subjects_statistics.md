# GET /api/config/subjects/statistics

## Summary
Get Subject Specific Statistics

## Description
Get subject-specific statistics and analysis data.

This endpoint provides detailed statistics about drawings and analyses
for specific subject categories or overall subject-related metrics.

## Tags
configuration

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| subject | query | unknown | No | No description |

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
GET /api/config/subjects/statistics
Content-Type: application/json
Accept: application/json
```

