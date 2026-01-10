# GET /api/metrics/usage

## Summary
Get Usage Metrics

## Description
Get comprehensive usage metrics for the dashboard.

Returns metrics including:
- Total analyses and drawings
- Time-based analysis counts (daily, weekly, monthly)
- Active user sessions and geographic distribution
- System health and performance metrics
- Processing time statistics

## Tags
metrics

## Parameters
No parameters required.

## Responses

### 200 - Successful Response

**application/json**:
```json
{}
```


## Complete Request Example

```http
GET /api/metrics/usage
Content-Type: application/json
Accept: application/json
```

