# GET /api/v1/documentation/metrics

## Summary
Get Documentation Metrics

## Description
Get comprehensive documentation metrics.

Returns metrics about documentation files, generation history,
success rates, and validation status.

## Tags
documentation

## Parameters
No parameters required.

## Responses

### 200 - Successful Response

**application/json**:
```json
{
  "total_files": 42,
  "last_generated": {},
  "generation_count": 42,
  "average_duration": 3.14,
  "success_rate": 3.14,
  "file_breakdown": {},
  "validation_status": {}
}
```


## Complete Request Example

```http
GET /api/v1/documentation/metrics
Content-Type: application/json
Accept: application/json
```

