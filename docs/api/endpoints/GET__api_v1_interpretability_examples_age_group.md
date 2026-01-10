# GET /api/v1/interpretability/examples/{age_group}

## Summary
Get Comparison Examples

## Description
Get comparison examples for educational purposes from a specific age group.

This endpoint provides examples of normal and anomalous drawings
to help users understand typical patterns and variations. Now supports
filtering by subject category for more targeted comparisons.

## Tags
interpretability

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| age_group | path | string | Yes | No description |
| example_type | query | string | No | No description |
| subject | query | unknown | No | No description |
| limit | query | integer | No | No description |

## Responses

### 200 - Successful Response

**application/json**:
```json
{
  "normal_examples": [
    {},
    {}
  ],
  "anomalous_examples": [
    {},
    {}
  ],
  "explanation_context": "example_string",
  "age_group": "example_string",
  "total_available": 42
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
GET /api/v1/interpretability/examples/{age_group}
Content-Type: application/json
Accept: application/json
```

