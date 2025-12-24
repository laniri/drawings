# User Journey - ExampleGallery

**Workflow ID**: `user_journey_examplegallery`
**Type**: User Journey
**Last Updated**: Unknown

## Overview

User journey workflow for ExampleGallery component

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | User Starts Journey |  |
| user_task_1 | userTask | Click  => setTypeFilter('all') | actor=user |
| user_task_2 | userTask | Click  => setTypeFilter('normal') | actor=user |
| user_task_3 | userTask | Click  => setTypeFilter('anomalous') | actor=user |
| user_task_4 | userTask | Click  => ExampleClick(example) | actor=user |
| user_task_5 | userTask | Click  => setTypeFilter('all') | actor=user |
| system_task_6 | serviceTask | Update SelectedExample | actor=system |
| system_task_7 | serviceTask | Update SelectedExample | actor=system |
| system_task_8 | serviceTask | Update TypeFilter | actor=system |
| system_task_9 | serviceTask | Update TypeFilter | actor=system |
| system_task_10 | serviceTask | Update TypeFilter | actor=system |
| end_11 | endEvent | Journey Complete |  |

## User Actions

- Click  => setTypeFilter('all')
- Click  => setTypeFilter('normal')
- Click  => setTypeFilter('anomalous')
- Click  => ExampleClick(example)
- Click  => setTypeFilter('all')

## System Responses

- Update SelectedExample
- Update SelectedExample
- Update TypeFilter
- Update TypeFilter
- Update TypeFilter

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `user_journey_examplegallery.bpmn`.

## Metadata

- **Elements Count**: 12
- **Workflow Type**: user_journey
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
