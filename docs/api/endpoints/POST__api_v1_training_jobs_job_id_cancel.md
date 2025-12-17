# POST /api/v1/training/jobs/{job_id}/cancel

## Summary
Cancel Training Job

## Description
Cancel a running training job.

This endpoint attempts to cancel a training job. For local jobs,
it stops the training process. For SageMaker jobs, it stops the
SageMaker training job.

## Parameters
- **job_id** (path): No description

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
POST /api/v1/training/jobs/{job_id}/cancel
```
