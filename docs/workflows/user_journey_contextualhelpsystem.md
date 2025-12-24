# User Journey - ContextualHelpSystem

**Workflow ID**: `user_journey_contextualhelpsystem`
**Type**: User Journey
**Last Updated**: Unknown

## Overview

User journey workflow for ContextualHelpSystem component

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | User Starts Journey |  |
| user_task_1 | userTask | Click Click | actor=user |
| user_task_2 | userTask | Click Click | actor=user |
| user_task_3 | userTask | Click Close | actor=user |
| user_task_4 | userTask | Click  => setShowAdvanced(!showAdvanced) | actor=user |
| user_task_5 | userTask | Click Close | actor=user |
| system_task_6 | serviceTask | Update AnchorEl | actor=system |
| system_task_7 | serviceTask | Update AnchorEl | actor=system |
| system_task_8 | serviceTask | Update ShowAdvanced | actor=system |
| system_task_9 | serviceTask | Update ShowAdvanced | actor=system |
| end_10 | endEvent | Journey Complete |  |

## User Actions

- Click Click
- Click Click
- Click Close
- Click  => setShowAdvanced(!showAdvanced)
- Click Close

## System Responses

- Update AnchorEl
- Update AnchorEl
- Update ShowAdvanced
- Update ShowAdvanced

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `user_journey_contextualhelpsystem.bpmn`.

## Metadata

- **Elements Count**: 11
- **Workflow Type**: user_journey
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
