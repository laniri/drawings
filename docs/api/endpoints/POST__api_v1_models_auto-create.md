# POST /api/v1/models/auto-create

## Summary
Auto Create Age Groups

## Description
Automatically create age group models based on data distribution.

This endpoint analyzes the available drawing data and creates
appropriate age group models with sufficient sample sizes.

## Parameters
- **force_recreate** (query): No description

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
POST /api/v1/models/auto-create
```
