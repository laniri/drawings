# GET /api/v1/analysis/drawing/{drawing_id}

## Summary
Get Drawing Analyses

## Description
Get all analyses for a specific drawing.

This endpoint returns the analysis history for a drawing,
ordered by most recent first.

## Tags
analysis

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| drawing_id | path | integer | Yes | No description |
| limit | query | integer | No | No description |

## Responses

### 200 - Successful Response

**application/json**:
```json
{
  "drawing_id": 42,
  "analyses": [
    {
      "id": 42,
      "drawing_id": 42,
      "anomaly_score": 3.14,
      "normalized_score": 3.14,
      "visual_anomaly_score": {},
      "subject_anomaly_score": {},
      "anomaly_attribution": {},
      "analysis_type": "example_string",
      "subject_category": {},
      "is_anomaly": true,
      "confidence": 3.14,
      "age_group": "example_string",
      "method_used": "example_string",
      "vision_model": "example_string",
      "analysis_timestamp": "example_string"
    },
    {
      "id": 42,
      "drawing_id": 42,
      "anomaly_score": 3.14,
      "normalized_score": 3.14,
      "visual_anomaly_score": {},
      "subject_anomaly_score": {},
      "anomaly_attribution": {},
      "analysis_type": "example_string",
      "subject_category": {},
      "is_anomaly": true,
      "confidence": 3.14,
      "age_group": "example_string",
      "method_used": "example_string",
      "vision_model": "example_string",
      "analysis_timestamp": "example_string"
    }
  ],
  "total_count": 42
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
GET /api/v1/analysis/drawing/{drawing_id}
Content-Type: application/json
Accept: application/json
```

