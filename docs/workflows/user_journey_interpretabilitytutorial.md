# User Journey - InterpretabilityTutorial

**Workflow ID**: `user_journey_interpretabilitytutorial`
**Type**: User Journey
**Last Updated**: Unknown

## Overview

User journey workflow for InterpretabilityTutorial component

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | User Starts Journey |  |
| user_task_1 | userTask | Click onClose | actor=user |
| user_task_2 | userTask | Click  => StepClick(index) | actor=user |
| user_task_3 | userTask | Click index === steps.length - 1 ? Complete : Next | actor=user |
| user_task_4 | userTask | Click Back | actor=user |
| user_task_5 | userTask | Click onClose | actor=user |
| system_task_6 | serviceTask | Update CompletedSteps | actor=system |
| system_task_7 | serviceTask | Update ActiveStep | actor=system |
| system_task_8 | serviceTask | Update ActiveStep | actor=system |
| system_task_9 | serviceTask | Update CompletedSteps | actor=system |
| system_task_10 | serviceTask | Update ActiveStep | actor=system |
| end_11 | endEvent | Journey Complete |  |

## User Actions

- Click onClose
- Click  => StepClick(index)
- Click index === steps.length - 1 ? Complete : Next
- Click Back
- Click onClose

## System Responses

- Update CompletedSteps
- Update ActiveStep
- Update ActiveStep
- Update CompletedSteps
- Update ActiveStep

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `user_journey_interpretabilitytutorial.bpmn`.

## Metadata

- **Elements Count**: 12
- **Workflow Type**: user_journey
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
