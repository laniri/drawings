# Technical Process - Score Normalizer

**Workflow ID**: `technical_process_score_normalizer`
**Type**: Technical Process
**Last Updated**: Unknown

## Overview

Technical process workflow for score_normalizer service

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | Process Start |  |
| service_task_1 | serviceTask | Get Score Normalizer | service=score_normalizer, method=get_score_normalizer |
| service_task_2 | serviceTask |   Init   | service=score_normalizer, method=__init__ |
| service_task_3 | serviceTask |  Get Age Group Statistics | service=score_normalizer, method=_get_age_group_statistics |
| service_task_4 | serviceTask | Normalize Score | service=score_normalizer, method=normalize_score |
| service_task_5 | serviceTask | Calculate Confidence | service=score_normalizer, method=calculate_confidence |
| end_6 | endEvent | Process Complete |  |

## Process Steps

- first
- all
- debug
- warning
- array
- error
- filter
- filter
- mean
- std
- min
- max
- median
- percentile
- percentile
- query
- query
- all
- array
- clip
- isfinite
- warning
- percentileofscore
- isfinite
- warning
- error
- filter
- query
- sum
- copy
- clip
- error
- _get_age_group_statistics
- get
- all
- array
- filter
- sum
- query
- sort
- find_appropriate_model
- append
- error
- warning
- normalize_score
- calculate_confidence
- all
- error
- filter
- _get_age_group_statistics
- append
- warning
- query
- clear_cache
- info
- clear
- info
- array
- zeros
- tolist
- std
- abs
- percentile
- percentile
- std
- mean

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `technical_process_score_normalizer.bpmn`.

## Metadata

- **Elements Count**: 7
- **Workflow Type**: technical_process
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
