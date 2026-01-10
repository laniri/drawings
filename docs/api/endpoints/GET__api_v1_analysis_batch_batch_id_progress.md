# GET /api/v1/analysis/batch/{batch_id}/progress

## Summary
Get Batch Progress

## Description
Get progress of batch analysis.

## Tags
analysis

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| batch_id | path | string | Yes | No description |

## Responses

### 200 - Successful Response

**application/json**:
```json
{
  "batch_id": "example_string",
  "total_drawings": 42,
  "completed": 42,
  "failed": 42,
  "status": "example_string",
  "results": [
    {
      "drawing": {
        "id": 42,
        "filename": "example_string",
        "age_years": 3.14,
        "subject": {},
        "expert_label": {},
        "drawing_tool": {},
        "prompt": {},
        "upload_timestamp": "example_string"
      },
      "analysis": {
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
      "interpretability": {},
      "comparison_examples": [
        {
          "drawing_id": 42,
          "filename": "example_string",
          "age_years": 3.14,
          "subject": {},
          "similarity_score": 3.14,
          "anomaly_score": 3.14,
          "normalized_score": 3.14
        },
        {
          "drawing_id": 42,
          "filename": "example_string",
          "age_years": 3.14,
          "subject": {},
          "similarity_score": 3.14,
          "anomaly_score": 3.14,
          "normalized_score": 3.14
        }
      ]
    },
    {
      "drawing": {
        "id": 42,
        "filename": "example_string",
        "age_years": 3.14,
        "subject": {},
        "expert_label": {},
        "drawing_tool": {},
        "prompt": {},
        "upload_timestamp": "example_string"
      },
      "analysis": {
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
      "interpretability": {},
      "comparison_examples": [
        {
          "drawing_id": 42,
          "filename": "example_string",
          "age_years": 3.14,
          "subject": {},
          "similarity_score": 3.14,
          "anomaly_score": 3.14,
          "normalized_score": 3.14
        },
        {
          "drawing_id": 42,
          "filename": "example_string",
          "age_years": 3.14,
          "subject": {},
          "similarity_score": 3.14,
          "anomaly_score": 3.14,
          "normalized_score": 3.14
        }
      ]
    }
  ],
  "errors": [
    {},
    {}
  ],
  "started_at": "example_string",
  "completed_at": {}
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
GET /api/v1/analysis/batch/{batch_id}/progress
Content-Type: application/json
Accept: application/json
```

