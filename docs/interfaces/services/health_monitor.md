# Health Monitor Service

Health monitoring service for system components and resources.

## Class: HealthStatus

Health status for a system component.

## Class: SystemMetrics

System resource metrics.

## Class: HealthMonitor

Service for monitoring system health and resources.

### get_metrics_history

Get metrics history for the specified number of hours.

**Signature**: `get_metrics_history(hours)`

### get_overall_status

Get overall system status based on all components.

**Signature**: `get_overall_status()`

### get_alerts

Get current alerts based on health checks.

**Signature**: `get_alerts()`

### update_alert_thresholds

Update alert thresholds.

**Signature**: `update_alert_thresholds(thresholds)`

