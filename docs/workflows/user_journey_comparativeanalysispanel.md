# User Journey - ComparativeAnalysisPanel

**Workflow ID**: `user_journey_comparativeanalysispanel`
**Type**: User Journey
**Last Updated**: Unknown

## Overview

User journey workflow for ComparativeAnalysisPanel component

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | User Starts Journey |  |
| user_task_1 | userTask | Click  => ExampleClick(example) | actor=user |
| user_task_2 | userTask | Click  => setDetailDialogOpen(false) | actor=user |
| user_task_3 | userTask | Click  => ViewExample(selectedExample.drawing_id) | actor=user |
| user_task_4 | userTask | Change (_, newValue) => setTabValue(newValue) | actor=user |
| system_task_5 | serviceTask | Update Loading | actor=system |
| system_task_6 | serviceTask | Update Error | actor=system |
| system_task_7 | serviceTask | Update ComparisonData | actor=system |
| system_task_8 | serviceTask | Update Error | actor=system |
| system_task_9 | serviceTask | Update Loading | actor=system |
| end_10 | endEvent | Journey Complete |  |

## User Actions

- Click  => ExampleClick(example)
- Click  => setDetailDialogOpen(false)
- Click  => ViewExample(selectedExample.drawing_id)
- Change (_, newValue) => setTabValue(newValue)

## System Responses

- Update Loading
- Update Error
- Update ComparisonData
- Update Error
- Update Loading

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `user_journey_comparativeanalysispanel.bpmn`.

## Metadata

- **Elements Count**: 11
- **Workflow Type**: user_journey
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
