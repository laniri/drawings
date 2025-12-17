# POST /api/v1/training/deploy

## Summary
Deploy Trained Model

## Description
Deploy trained model parameters to production system.

This endpoint loads trained model parameters and creates a new
age group model for production use.

## Parameters
No parameters

## Request Body
Request body required

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
POST /api/v1/training/deploy
```
