# GET /api/interpretability/{analysis_id}/confidence

## Summary
Get Confidence Metrics

## Description
Get confidence metrics and reliability scores for interpretability results.

This endpoint provides detailed confidence information to help users
assess the trustworthiness of the analysis and interpretations.

## Tags
interpretability

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| analysis_id | path | integer | Yes | No description |

## Responses

### 200 - Successful Response

**application/json**:
```json
{
  "overall_confidence": 3.14,
  "explanation_reliability": 3.14,
  "model_certainty": 3.14,
  "data_sufficiency": "example_string",
  "warnings": [
    "example_string",
    "example_string"
  ],
  "technical_details": {}
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
GET /api/interpretability/{analysis_id}/confidence
Content-Type: application/json
Accept: application/json
```

