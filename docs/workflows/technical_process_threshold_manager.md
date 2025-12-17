# Technical Process - Threshold Manager

**Workflow ID**: `technical_process_threshold_manager`
**Type**: Technical Process
**Last Updated**: Unknown

## Overview

Technical process workflow for threshold_manager service

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | Process Start |  |
| service_task_1 | serviceTask | Get Threshold Manager | service=threshold_manager, method=get_threshold_manager |
| service_task_2 | serviceTask |   Post Init   | service=threshold_manager, method=__post_init__ |
| service_task_3 | serviceTask |   Init   | service=threshold_manager, method=__init__ |
| service_task_4 | serviceTask | Calculate Percentile Threshold | service=threshold_manager, method=calculate_percentile_threshold |
| service_task_5 | serviceTask | Calculate Model Threshold | service=threshold_manager, method=calculate_model_threshold |
| end_6 | endEvent | Process Complete |  |

## Process Steps

- percentile
- info
- isfinite
- warning
- isfinite
- first
- all
- array
- info
- calculate_percentile_threshold
- calculate_percentile_threshold
- error
- filter
- filter
- mean
- std
- min
- max
- median
- query
- query
- first
- commit
- info
- isfinite
- error
- rollback
- filter
- keys
- startswith
- query
- all
- info
- set_current_percentile
- filter
- calculate_model_threshold
- update_model_threshold
- append
- append
- error
- query
- find_appropriate_model
- get_threshold_for_age
- find_appropriate_model
- warning
- all
- mean
- std
- filter
- query
- clear
- info

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `technical_process_threshold_manager.bpmn`.

## Metadata

- **Elements Count**: 7
- **Workflow Type**: technical_process
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
