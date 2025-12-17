# Technical Process - Data Sufficiency Service

**Workflow ID**: `technical_process_data_sufficiency_service`
**Type**: Technical Process
**Last Updated**: Unknown

## Overview

Technical process workflow for data_sufficiency_service service

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | Process Start |  |
| service_task_1 | serviceTask | Get Data Sufficiency Analyzer | service=data_sufficiency_service, method=get_data_sufficiency_analyzer |
| service_task_2 | serviceTask | Get Data Augmentation Suggester | service=data_sufficiency_service, method=get_data_augmentation_suggester |
| service_task_3 | serviceTask | To Dict | service=data_sufficiency_service, method=to_dict |
| service_task_4 | serviceTask | To Dict | service=data_sufficiency_service, method=to_dict |
| service_task_5 | serviceTask | To Dict | service=data_sufficiency_service, method=to_dict |
| end_6 | endEvent | Process Complete |  |

## Process Steps

- info
- info
- all
- _calculate_data_quality_score
- _analyze_subject_distribution
- error
- filter
- query
- get
- info
- analyze_age_group_data
- error
- append
- append
- values
- values
- append
- append
- append
- sort
- info
- analyze_age_group_data
- append
- error
- _are_groups_mergeable
- append
- add
- _calculate_improvement_score
- _generate_merging_rationale
- append
- info
- extend
- extend
- items
- extend
- values
- extend
- append

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `technical_process_data_sufficiency_service.bpmn`.

## Metadata

- **Elements Count**: 7
- **Workflow Type**: technical_process
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
