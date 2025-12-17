# POST /api/v1/training/jobs

## Summary
Submit Training Job

## Description
Submit a new training job to either local or SageMaker environment.

This endpoint creates and submits a training job based on the specified
environment. For SageMaker jobs, it handles container building, data upload,
and job submission. For local jobs, it starts training immediately.

## Parameters
No parameters

## Request Body
Request body required

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
POST /api/v1/training/jobs
```
