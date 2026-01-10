# GET /api/v1/metrics/usage

**⚠️ UPDATED January 2026**: This endpoint now returns a nested data structure. See [detailed documentation](./metrics-usage.md) for complete schema and examples.

## Summary
Get Usage Metrics

## Description
Get comprehensive usage metrics for the dashboard with nested data structure.

Returns metrics including:
- **Database statistics**: Total analyses, drawings, anomaly counts
- **Time-based analysis counts**: Daily, weekly, monthly
- **Active user sessions**: Session tracking and geographic distribution
- **System health metrics**: Uptime, error rates, resource usage
- **Processing time statistics**: Average processing times

## Tags
metrics

## Parameters
No parameters required.

## Response Structure

The response contains a nested structure with the following top-level keys:
- `timestamp`: ISO 8601 timestamp
- `database`: Database-related statistics
- `time_based`: Time-based analysis counts
- `sessions`: User session metrics
- `system_health`: System health and performance metrics
- `geographic`: Geographic distribution of sessions
- `uptime_seconds`: System uptime
- `last_updated`: Last update timestamp

## Responses

### 200 - Successful Response

**application/json**:
```json
{
  "status": "success",
  "data": {
    "timestamp": "2026-01-10T09:00:00.000000+00:00",
    "database": {
      "total_drawings": 37778,
      "total_analyses": 15234,
      "anomaly_count": 1523,
      "normal_count": 13711,
      "recent_analyses_count": 45,
      "age_groups_count": 8
    },
    "time_based": {
      "daily_analyses": 45,
      "weekly_analyses": 312,
      "monthly_analyses": 1456
    },
    "sessions": {
      "active_sessions": 3,
      "total_page_views": 127,
      "total_session_analyses": 15234
    },
    "system_health": {
      "uptime_seconds": 86400,
      "uptime_percentage": 99.9,
      "total_requests": 5432,
      "successful_requests": 5398,
      "failed_requests": 34,
      "error_rate": 0.0063,
      "average_response_time": 0.234,
      "memory_usage_mb": 512.45,
      "cpu_usage_percent": 23.5,
      "average_processing_time": 1.23
    },
    "geographic": {
      "United States": 2,
      "Canada": 1
    },
    "uptime_seconds": 86400,
    "last_updated": "2026-01-10T09:00:00.000000+00:00"
  }
}
```

### 500 - Internal Server Error

```json
{
  "detail": "Failed to retrieve usage metrics: <error message>"
}
```

## Complete Request Example

```http
GET /api/v1/metrics/usage
Content-Type: application/json
Accept: application/json
```

## Frontend Integration

This endpoint is consumed by the `UsageMetricsPanel` component with automatic refresh every 30 seconds.

## Related Documentation

- [Detailed Usage Metrics Documentation](./metrics-usage.md) - Complete schema, TypeScript interfaces, and examples
- [GET /api/v1/metrics/health](./GET__api_v1_metrics_health.md) - System health metrics
- [GET /api/v1/metrics/performance](./GET__api_v1_metrics_performance.md) - Performance metrics

## Version History

- **v2.0.0 (January 2026)**: Changed to nested data structure
- **v1.0.0**: Initial flat structure

