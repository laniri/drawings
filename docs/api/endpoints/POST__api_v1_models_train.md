# POST /api/v1/models/train

## Summary
Train Age Group Model

## Description
Train new age group model.

This endpoint starts training a new autoencoder model for the specified
age range. Training is performed in the background and progress can be
tracked using the returned job ID.

## Parameters
No parameters

## Request Body
Request body required

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
POST /api/v1/models/train
```
