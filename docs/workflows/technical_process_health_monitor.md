# Technical Process - Health Monitor

**Workflow ID**: `technical_process_health_monitor`
**Type**: Technical Process
**Last Updated**: Unknown

## Overview

Technical process workflow for health_monitor service

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | Process Start |  |
| service_task_1 | serviceTask |   Init   | service=health_monitor, method=__init__ |
| service_task_2 | serviceTask |  Add Metrics To History | service=health_monitor, method=_add_metrics_to_history |
| service_task_3 | serviceTask | Get Metrics History | service=health_monitor, method=get_metrics_history |
| service_task_4 | serviceTask | Get Overall Status | service=health_monitor, method=get_overall_status |
| service_task_5 | serviceTask | Get Alerts | service=health_monitor, method=get_alerts |
| end_6 | endEvent | Process Complete |  |

## Process Steps

- append
- utcnow
- values
- values
- append
- isoformat
- items
- info
- warning

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `technical_process_health_monitor.bpmn`.

## Metadata

- **Elements Count**: 7
- **Workflow Type**: technical_process
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
