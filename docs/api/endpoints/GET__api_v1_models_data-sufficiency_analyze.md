# GET /api/v1/models/data-sufficiency/analyze

## Summary
Analyze Data Sufficiency

## Description
Analyze data sufficiency for age groups.

This endpoint analyzes the available data for specified age groups
and provides warnings about insufficient data, unbalanced distributions,
and other data quality issues.

Args:
    age_groups: Comma-separated list of age ranges (e.g., "3-4,4-5,5-6")
               If not provided, analyzes all existing age group models

## Parameters
- **age_groups** (query): No description

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
GET /api/v1/models/data-sufficiency/analyze
```
