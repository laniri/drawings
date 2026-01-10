# GET /api/config/

## Summary
Get Config

## Description
Get current system configuration.

This endpoint returns the current system configuration including
model settings, threshold parameters, and age grouping strategy.

## Tags
configuration

## Parameters
No parameters required.

## Responses

### 200 - Successful Response

**application/json**:
```json
{
  "vision_model": "example_string",
  "anomaly_detection_method": "example_string",
  "threshold_percentile": 3.14,
  "age_grouping_strategy": "example_string",
  "min_samples_per_group": 42,
  "max_age_group_span": 3.14
}
```


## Complete Request Example

```http
GET /api/config/
Content-Type: application/json
Accept: application/json
```

