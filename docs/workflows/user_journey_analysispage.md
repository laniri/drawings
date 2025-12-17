# User Journey - AnalysisPage

**Workflow ID**: `user_journey_analysispage`
**Type**: User Journey
**Last Updated**: Unknown

## Overview

User journey workflow for AnalysisPage component

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | User Starts Journey |  |
| user_task_1 | userTask | Click  => analysisMutation.mutate | actor=user |
| user_task_2 | userTask | Click  => analysisMutation.mutate | actor=user |
| user_task_3 | userTask | Click  => analysisMutation.mutate | actor=user |
| user_task_4 | userTask | Change (_, newValue) => setActiveTab(newValue) | actor=user |
| user_task_5 | userTask | Change (e) => setShowSaliency(e.target.checked) | actor=user |
| system_task_6 | serviceTask | Update ActiveTab | actor=system |
| system_task_7 | serviceTask | Update ShowSaliency | actor=system |
| system_task_8 | serviceTask | Update SaliencyOpacity | actor=system |
| end_9 | endEvent | Journey Complete |  |

## User Actions

- Click  => analysisMutation.mutate
- Click  => analysisMutation.mutate
- Click  => analysisMutation.mutate
- Change (_, newValue) => setActiveTab(newValue)
- Change (e) => setShowSaliency(e.target.checked)

## System Responses

- Update ActiveTab
- Update ShowSaliency
- Update SaliencyOpacity

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `user_journey_analysispage.bpmn`.

## Metadata

- **Elements Count**: 10
- **Workflow Type**: user_journey
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
