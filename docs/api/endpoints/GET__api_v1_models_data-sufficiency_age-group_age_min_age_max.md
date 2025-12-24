# GET /api/v1/models/data-sufficiency/age-group/{age_min}/{age_max}

## Summary
Analyze Specific Age Group

## Description
Analyze data sufficiency for a specific age group.

This endpoint provides detailed analysis of data availability,
quality, and distribution for a single age group.

**Error Handling**: Invalid age ranges (age_min >= age_max) are handled gracefully and return valid responses with zero samples rather than errors.

## Parameters
- **age_min** (path): Minimum age for the group (float)
- **age_max** (path): Maximum age for the group (float)

## Responses
- **200**: Successful Response - Returns AgeGroupDataInfo with analysis results
- **422**: Validation Error - Only for malformed path parameters

## Response Schema (200)
```json
{
  "age_min": 5.0,
  "age_max": 6.0,
  "sample_count": 150,
  "is_sufficient": true,
  "recommended_min_samples": 100,
  "data_quality_score": 0.85,
  "subjects_distribution": {
    "person": 75,
    "house": 45,
    "tree": 30
  },
  "age_distribution": [5.1, 5.2, 5.3, ...]
}
```

## Edge Cases
- **Invalid age ranges** (age_min >= age_max): Returns valid response with sample_count=0
- **Out-of-bounds ages**: Returns valid response with sample_count=0
- **No data available**: Returns valid response with sample_count=0

## Example
```http
GET /api/v1/models/data-sufficiency/age-group/5.0/6.0
```
