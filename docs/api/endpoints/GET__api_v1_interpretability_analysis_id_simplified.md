# GET /api/v1/interpretability/{analysis_id}/simplified

## Summary
Get Simplified Explanation

## Description
Get simplified, non-technical explanations suitable for educators and parents.

This endpoint provides explanations adapted for different user roles
with accessible language and clear recommendations.

## Tags
interpretability

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| analysis_id | path | integer | Yes | No description |
| user_role | query | unknown | No | No description |

## Responses

### 200 - Successful Response

**application/json**:
```json
{
  "summary": "example_string",
  "key_findings": [
    "example_string",
    "example_string"
  ],
  "visual_indicators": [
    {},
    {}
  ],
  "confidence_level": "example_string",
  "age_appropriate_context": "example_string",
  "recommendations": [
    "example_string",
    "example_string"
  ]
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
GET /api/v1/interpretability/{analysis_id}/simplified
Content-Type: application/json
Accept: application/json
```

