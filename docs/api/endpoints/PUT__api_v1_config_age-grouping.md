# PUT /api/v1/config/age-grouping

## Summary
Update Age Grouping

## Description
Modify age grouping strategy.

This endpoint updates the age grouping configuration and can
optionally trigger recreation of age group models.

## Parameters
No parameters

## Request Body
Request body required

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
PUT /api/v1/config/age-grouping
```
