# POST /api/v1/training/jobs

## Summary
Submit Training Job

## Description
Submit a new training job to either local or SageMaker environment.

This endpoint creates and submits a training job based on the specified
environment. For SageMaker jobs, it handles container building, data upload,
and job submission. For local jobs, it starts training immediately.

## Tags
training

## Parameters
No parameters required.

## Request Body
Request body required

### Request Body Examples

**application/json**:
```json
{
  "job_name": "example_string",
  "environment": "example_string",
  "dataset_folder": "example_string",
  "metadata_file": "example_string",
  "learning_rate": 3.14,
  "batch_size": 42,
  "epochs": 42,
  "train_split": 3.14,
  "validation_split": 3.14,
  "test_split": 3.14,
  "instance_type": {},
  "instance_count": 42
}
```


## Responses

### 200 - Successful Response

**application/json**:
```json
{
  "id": 42,
  "job_name": "example_string",
  "environment": "example_string",
  "status": "example_string",
  "start_timestamp": {},
  "end_timestamp": {},
  "sagemaker_job_arn": {}
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
POST /api/v1/training/jobs
Content-Type: application/json
Accept: application/json
```

