# GET /api/v1/drawings/

## Summary
List Drawings

## Description
List drawings with optional filtering and pagination.

## Parameters
- **age_min** (query): No description
- **age_max** (query): No description
- **subject** (query): No description
- **expert_label** (query): No description
- **page** (query): No description
- **page_size** (query): No description

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
GET /api/v1/drawings/
```
