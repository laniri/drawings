# GET /api/training/jobs

## Summary
List Training Jobs

## Description
List training jobs with optional filtering.

This endpoint returns a list of training jobs, optionally filtered
by environment (local/sagemaker) and status.

## Tags
training

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| environment | query | unknown | No | No description |
| status | query | unknown | No | No description |
| limit | query | integer | No | No description |

## Responses

### 200 - Successful Response

**application/json**:
```json
[
  {
    "id": 42,
    "job_name": "example_string",
    "environment": "example_string",
    "status": "example_string",
    "start_timestamp": {},
    "end_timestamp": {},
    "sagemaker_job_arn": {}
  },
  {
    "id": 42,
    "job_name": "example_string",
    "environment": "example_string",
    "status": "example_string",
    "start_timestamp": {},
    "end_timestamp": {},
    "sagemaker_job_arn": {}
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
GET /api/training/jobs
Content-Type: application/json
Accept: application/json
```

