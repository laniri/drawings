# POST /api/v1/training/sagemaker/setup

## Summary
Setup Sagemaker Environment

## Description
Setup SageMaker training environment.

This endpoint helps set up the necessary AWS resources for
SageMaker training, including IAM roles and container repositories.

## Tags
training

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| s3_bucket | query | string | Yes | No description |
| ecr_repository | query | unknown | No | No description |

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
POST /api/v1/training/sagemaker/setup
Content-Type: application/json
Accept: application/json
```

