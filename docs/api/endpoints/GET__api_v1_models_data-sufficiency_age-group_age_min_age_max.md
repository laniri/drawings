# GET /api/v1/models/data-sufficiency/age-group/{age_min}/{age_max}

## Summary
Analyze Specific Age Group

## Description
Analyze data sufficiency for a specific age group.

This endpoint provides detailed analysis of data availability,
quality, and distribution for a single age group.

## Tags
models

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| age_min | path | number | Yes | No description |
| age_max | path | number | Yes | No description |

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
GET /api/v1/models/data-sufficiency/age-group/{age_min}/{age_max}
Content-Type: application/json
Accept: application/json
```

