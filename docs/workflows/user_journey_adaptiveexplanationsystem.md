# User Journey - AdaptiveExplanationSystem

**Workflow ID**: `user_journey_adaptiveexplanationsystem`
**Type**: User Journey
**Last Updated**: Unknown

## Overview

User journey workflow for AdaptiveExplanationSystem component

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | User Starts Journey |  |
| user_task_1 | userTask | Click  => {
                // Could open advanced settings dialog
               | actor=user |
| user_task_2 | userTask | Change (e) => setAutoAdapt(e.target.checked) | actor=user |
| user_task_3 | userTask | Change (e) => ConfigChange('userRole', e.target.value) | actor=user |
| user_task_4 | userTask | Change (_, value) => ConfigChange('complexity', value) | actor=user |
| user_task_5 | userTask | Change (e) => ConfigChange('explanationStyle', e.target.value) | actor=user |
| system_task_6 | serviceTask | Update Config | actor=system |
| system_task_7 | serviceTask | Update AdaptiveContent | actor=system |
| system_task_8 | serviceTask | Update Config | actor=system |
| system_task_9 | serviceTask | Update AutoAdapt | actor=system |
| end_10 | endEvent | Journey Complete |  |

## User Actions

- Click  => {
                // Could open advanced settings dialog
              
- Change (e) => setAutoAdapt(e.target.checked)
- Change (e) => ConfigChange('userRole', e.target.value)
- Change (_, value) => ConfigChange('complexity', value)
- Change (e) => ConfigChange('explanationStyle', e.target.value)

## System Responses

- Update Config
- Update AdaptiveContent
- Update Config
- Update AutoAdapt

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `user_journey_adaptiveexplanationsystem.bpmn`.

## Metadata

- **Elements Count**: 11
- **Workflow Type**: user_journey
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
