# Health Monitor Contract

## Overview
Service contract for Health Monitor (service)

**Source File**: `app/services/health_monitor.py`

## Interface Specification

### Classes

#### HealthStatus

Health status for a system component.

**Attributes**:

- `name: str`
- `status: str`
- `message: str`
- `details: Dict[str, Any]`
- `last_check: datetime`
- `response_time_ms: Optional[float]`

#### SystemMetrics

System resource metrics.

**Attributes**:

- `timestamp: datetime`
- `cpu_percent: float`
- `memory_percent: float`
- `memory_available_gb: float`
- `memory_total_gb: float`
- `disk_percent: float`
- `disk_free_gb: float`
- `disk_total_gb: float`
- `active_connections: int`
- `process_count: int`

#### HealthMonitor

Service for monitoring system health and resources.

## Methods

### get_metrics_history

Get metrics history for the specified number of hours.

**Signature**: `get_metrics_history(hours: int) -> <ast.Subscript object at 0x1104bdd10>`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `hours` | `int` | Parameter description |

**Returns**: `<ast.Subscript object at 0x1104bdd10>`

### get_overall_status

Get overall system status based on all components.

**Signature**: `get_overall_status() -> str`

**Returns**: `str`

### get_alerts

Get current alerts based on health checks.

**Signature**: `get_alerts() -> <ast.Subscript object at 0x1105ae9d0>`

**Returns**: `<ast.Subscript object at 0x1105ae9d0>`

### update_alert_thresholds

Update alert thresholds.

**Signature**: `update_alert_thresholds(thresholds: Dict[str, float])`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `thresholds` | `Dict[str, float]` | Parameter description |

## Defined Interfaces

### HealthMonitorInterface

**Type**: Protocol
**Implemented by**: HealthMonitor

**Methods**:

- `get_metrics_history(hours: int) -> <ast.Subscript object at 0x1104bdd10>`
- `get_overall_status() -> str`
- `get_alerts() -> <ast.Subscript object at 0x1105ae9d0>`
- `update_alert_thresholds(thresholds: Dict[str, float])`

## Usage Examples

```python
# Example usage of the service contract
# TODO: Add specific usage examples
```

## Validation

This contract is automatically validated against the implementation in:
- Source file: `app/services/health_monitor.py`
- Last validated: 2025-12-16 15:47:04

