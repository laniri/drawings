# Technical Process - Comparison Service

**Workflow ID**: `technical_process_comparison_service`
**Type**: Technical Process
**Last Updated**: Unknown

## Overview

Technical process workflow for comparison_service service

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | Process Start |  |
| service_task_1 | serviceTask | Get Comparison Service | service=comparison_service, method=get_comparison_service |
| service_task_2 | serviceTask |   Init   | service=comparison_service, method=__init__ |
| service_task_3 | serviceTask | Find Similar Normal Examples | service=comparison_service, method=find_similar_normal_examples |
| service_task_4 | serviceTask |  Get Drawing Embedding | service=comparison_service, method=_get_drawing_embedding |
| service_task_5 | serviceTask |  Get Normal Drawings In Age Group | service=comparison_service, method=_get_normal_drawings_in_age_group |
| end_6 | endEvent | Process Complete |  |

## Process Steps

- _get_drawing_embedding
- _get_normal_drawings_in_age_group
- items
- sort
- warning
- info
- _get_drawing_embedding
- _calculate_cosine_similarity
- error
- append
- first
- first
- retrieve_embedding
- error
- order_by
- filter
- desc
- filter
- query
- query
- limit
- all
- error
- order_by
- asc
- filter
- join
- join
- query
- norm
- norm
- dot
- error
- count
- count
- count
- error
- filter
- distinct
- distinct
- query
- filter
- filter
- join
- join
- join
- query
- query

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `technical_process_comparison_service.bpmn`.

## Metadata

- **Elements Count**: 7
- **Workflow Type**: technical_process
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
