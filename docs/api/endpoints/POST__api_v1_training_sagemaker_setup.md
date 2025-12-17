# POST /api/v1/training/sagemaker/setup

## Summary
Setup Sagemaker Environment

## Description
Setup SageMaker training environment.

This endpoint helps set up the necessary AWS resources for
SageMaker training, including IAM roles and container repositories.

## Parameters
- **s3_bucket** (query): No description
- **ecr_repository** (query): No description

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
POST /api/v1/training/sagemaker/setup
```
