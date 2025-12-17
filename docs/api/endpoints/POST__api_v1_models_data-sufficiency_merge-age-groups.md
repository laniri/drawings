# POST /api/v1/models/data-sufficiency/merge-age-groups

## Summary
Merge Age Groups

## Description
Merge age groups to improve data sufficiency.

This endpoint deactivates the original age group models and creates
a new merged age group model with combined data.

## Parameters
No parameters

## Request Body
Request body required

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
POST /api/v1/models/data-sufficiency/merge-age-groups
```
