# PUT /api/v1/models/{model_id}/threshold

## Summary
Update Model Threshold

## Description
Update model threshold.

This endpoint allows updating the anomaly detection threshold
for a specific age group model. The threshold can be set directly
or calculated from a percentile of validation data.

## Parameters
- **model_id** (path): No description

## Request Body
Request body required

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
PUT /api/v1/models/{model_id}/threshold
```
