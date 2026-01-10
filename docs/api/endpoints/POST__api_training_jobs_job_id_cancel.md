# POST /api/training/jobs/{job_id}/cancel

## Summary
Cancel Training Job

## Description
Cancel a running training job.

This endpoint attempts to cancel a training job. For local jobs,
it stops the training process. For SageMaker jobs, it stops the
SageMaker training job.

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
POST /api/training/jobs/{job_id}/cancel
Content-Type: application/json
Accept: application/json
```

