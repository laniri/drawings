# PUT /api/v1/config/age-grouping

## Summary
Update Age Grouping

## Description
Modify age grouping strategy.

This endpoint updates the age grouping configuration and can
optionally trigger recreation of age group models.

## Tags
configuration

## Parameters
No parameters required.

## Request Body
Request body required

### Request Body Examples

**application/json**:
```json
{
  "threshold_percentile": {},
  "age_grouping_strategy": {},
  "min_samples_per_group": {},
  "max_age_group_span": {}
}
```


## Responses

### 200 - Successful Response

**application/json**:
```json
{
  "success": true,
  "message": "example_string",
  "data": {}
}
```

### 422 - Validation Error

**application/json**:
```json
{
  "detail": [
    {
      "loc": [
        {},
        {}
      ],
      "msg": "example_string",
      "type": "example_string"
    },
    {
      "loc": [
        {},
        {}
      ],
      "msg": "example_string",
      "type": "example_string"
    }
  ]
}
```


## Complete Request Example

```http
PUT /api/v1/config/age-grouping
Content-Type: application/json
Accept: application/json
```

