# POST /api/v1/documentation/batch/generate

## Summary
Batch Generate Documentation

## Description
Batch generate multiple documentation categories with scheduling.

Allows generating multiple categories in sequence with different
configurations for each category.

## Parameters
No parameters

## Request Body
Request body required

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
POST /api/v1/documentation/batch/generate
```
