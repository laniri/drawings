# GET /api/v1/models/data-sufficiency/warnings

## Summary
Get Data Warnings

## Description
Get data sufficiency warnings for all age groups.

This endpoint returns warnings about data quality issues,
optionally filtered by severity level.

## Parameters
- **severity** (query): No description

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
GET /api/v1/models/data-sufficiency/warnings
```
