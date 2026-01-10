# GET /api/v1/training/jobs/{job_id}/reports

## Summary
Get Training Reports

## Description
Get training reports for a specific job.

This endpoint returns all training reports associated with a job,
including metrics, model paths, and performance summaries.

## Tags
training

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| job_id | path | integer | Yes | No description |

## Responses

### 200 - Successful Response

**application/json**:
```json
[
  {
    "id": 42,
    "final_loss": 3.14,
    "validation_accuracy": 3.14,
    "best_epoch": 42,
    "training_time_seconds": 3.14,
    "model_parameters_path": "example_string",
    "report_file_path": "example_string",
    "created_timestamp": "example_string"
  },
  {
    "id": 42,
    "final_loss": 3.14,
    "validation_accuracy": 3.14,
    "best_epoch": 42,
    "training_time_seconds": 3.14,
    "model_parameters_path": "example_string",
    "report_file_path": "example_string",
    "created_timestamp": "example_string"
  }
]
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
GET /api/v1/training/jobs/{job_id}/reports
Content-Type: application/json
Accept: application/json
```

