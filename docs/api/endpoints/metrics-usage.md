# Usage Metrics API Endpoint

## Endpoint

`GET /api/v1/metrics/usage`

## Description

Get comprehensive usage metrics for the dashboard with nested data structure including database statistics, time-based analysis counts, active sessions, system health, and geographic distribution.

## Authentication

No authentication required (public endpoint).

## Request

No request parameters required.

## Response

### Success Response (200 OK)

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

### Response Schema

#### Top Level
- `status` (string): Response status, always "success"
- `data` (object): Nested metrics data

#### Data Object
- `timestamp` (string, ISO 8601): Timestamp when metrics were collected
- `database` (object): Database-related statistics
- `time_based` (object): Time-based analysis counts
- `sessions` (object): User session metrics
- `system_health` (object): System health and performance metrics
- `geographic` (object): Geographic distribution of sessions
- `uptime_seconds` (integer): System uptime in seconds (duplicate for backward compatibility)
- `last_updated` (string, ISO 8601): Last update timestamp

#### Database Object
- `total_drawings` (integer): Total number of drawings in database
- `total_analyses` (integer): Total number of analyses performed
- `anomaly_count` (integer): Number of analyses classified as anomalies
- `normal_count` (integer): Number of analyses classified as normal
- `recent_analyses_count` (integer): Number of analyses in last 24 hours
- `age_groups_count` (integer): Number of distinct age groups

#### Time-Based Object
- `daily_analyses` (integer): Analyses performed today
- `weekly_analyses` (integer): Analyses performed in last 7 days
- `monthly_analyses` (integer): Analyses performed in last 30 days

#### Sessions Object
- `active_sessions` (integer): Currently active user sessions
- `total_page_views` (integer): Total page views across all active sessions
- `total_session_analyses` (integer): Total analyses across all sessions

#### System Health Object
- `uptime_seconds` (integer): System uptime in seconds
- `uptime_percentage` (float): Uptime percentage (0-100)
- `total_requests` (integer): Total HTTP requests processed
- `successful_requests` (integer): Successful HTTP requests
- `failed_requests` (integer): Failed HTTP requests
- `error_rate` (float): Error rate (0-1)
- `average_response_time` (float): Average response time in seconds
- `memory_usage_mb` (float): Current memory usage in MB
- `cpu_usage_percent` (float): Current CPU usage percentage (0-100)
- `average_processing_time` (float): Average analysis processing time in seconds

#### Geographic Object
Key-value pairs where:
- Key (string): Country name or "Unknown"
- Value (integer): Number of active sessions from that location

## Error Response

### 500 Internal Server Error

```json
{
  "detail": "Failed to retrieve usage metrics: <error message>"
}
```

## Usage Example

### cURL

```bash
curl -X GET "https://api.example.com/api/v1/metrics/usage" \
  -H "accept: application/json"
```

### Python

```python
import requests

response = requests.get("https://api.example.com/api/v1/metrics/usage")
data = response.json()

# Access nested data
total_analyses = data["data"]["database"]["total_analyses"]
daily_count = data["data"]["time_based"]["daily_analyses"]
error_rate = data["data"]["system_health"]["error_rate"]
```

### JavaScript/TypeScript

```typescript
const response = await fetch('/api/v1/metrics/usage');
const result = await response.json();

// Access nested data
const totalAnalyses = result.data.database.total_analyses;
const dailyCount = result.data.time_based.daily_analyses;
const errorRate = result.data.system_health.error_rate;
```

## Frontend Integration

This endpoint is consumed by the `UsageMetricsPanel` component in the React frontend:

```typescript
interface UsageMetrics {
  timestamp: string;
  database: {
    total_drawings: number;
    total_analyses: number;
    anomaly_count: number;
    normal_count: number;
    recent_analyses_count: number;
    age_groups_count: number;
  };
  time_based: {
    daily_analyses: number;
    weekly_analyses: number;
    monthly_analyses: number;
  };
  sessions: {
    active_sessions: number;
    total_page_views: number;
    total_session_analyses: number;
  };
  system_health: {
    uptime_seconds: number;
    uptime_percentage: number;
    total_requests: number;
    successful_requests: number;
    failed_requests: number;
    error_rate: number;
    average_response_time: number;
    memory_usage_mb: number;
    cpu_usage_percent: number;
    average_processing_time: number;
  };
  geographic: Record<string, number>;
  uptime_seconds: number;
}
```

## Notes

- Metrics are calculated in real-time from in-memory data structures
- Geographic distribution requires session tracking to be enabled
- CloudWatch integration is optional; local mode provides estimates
- The endpoint automatically refreshes every 30 seconds in the frontend
- Database statistics (`total_drawings`, `age_groups_count`) may return 0 if not tracked by the metrics service

## Related Endpoints

- `GET /api/v1/metrics/health` - System health metrics
- `GET /api/v1/metrics/sessions` - Session-specific metrics
- `GET /api/v1/metrics/performance` - Detailed performance metrics

## Version History

- **v2.0.0 (January 2026)**: Changed to nested data structure with `database`, `time_based`, `sessions`, and `system_health` objects
- **v1.0.0**: Initial flat structure with top-level fields
