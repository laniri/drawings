# GET /api/v1/training/jobs/{job_id}

## Summary
Get Training Job Status

## Description
Get detailed status of a specific training job.

This endpoint returns comprehensive information about a training job,
including progress, metrics, and environment-specific details.

## Parameters
- **job_id** (path): No description

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
GET /api/v1/training/jobs/{job_id}
```
