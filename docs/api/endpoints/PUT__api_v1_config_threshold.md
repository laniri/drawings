# PUT /api/v1/config/threshold

## Summary
Update Threshold Settings

## Description
Update global threshold settings.

This endpoint recalculates thresholds for all active models
using the specified percentile value from the request body.

## Parameters
No parameters

## Request Body
Request body required

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
PUT /api/v1/config/threshold
```
