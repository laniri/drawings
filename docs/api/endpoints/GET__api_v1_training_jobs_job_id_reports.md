# GET /api/v1/training/jobs/{job_id}/reports

## Summary
Get Training Reports

## Description
Get training reports for a specific job.

This endpoint returns all training reports associated with a job,
including metrics, model paths, and performance summaries.

## Parameters
- **job_id** (path): No description

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
GET /api/v1/training/jobs/{job_id}/reports
```
