# GET /api/models/data-sufficiency/analyze

## Summary
Analyze Data Sufficiency

## Description
Analyze data sufficiency for age groups.

This endpoint analyzes the available data for specified age groups
and provides warnings about insufficient data, unbalanced distributions,
and other data quality issues.

Args:
    age_groups: Comma-separated list of age ranges (e.g., "3-4,4-5,5-6")
               If not provided, analyzes all existing age group models

## Tags
models

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| age_groups | query | unknown | No | No description |

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
GET /api/models/data-sufficiency/analyze
Content-Type: application/json
Accept: application/json
```

