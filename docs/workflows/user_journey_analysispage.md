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
| user_task_4 | userTask | Click (_, explanation) => {
                          setSelectedRegionExplanation(explanation)
                         | actor=user |
| user_task_5 | userTask | Change (_, newValue) => setActiveTab(newValue) | actor=user |
| system_task_6 | serviceTask | Update ActiveTab | actor=system |
| system_task_7 | serviceTask | Update SelectedRegionExplanation | actor=system |
| system_task_8 | serviceTask | Update CurrentExplanationLevel | actor=system |
| system_task_9 | serviceTask | Update SelectedRegionExplanation | actor=system |
| system_task_10 | serviceTask | Update ShowSaliency | actor=system |
| end_11 | endEvent | Journey Complete |  |

## User Actions

- Click  => analysisMutation.mutate
- Click  => analysisMutation.mutate
- Click  => analysisMutation.mutate
- Click (_, explanation) => {
                          setSelectedRegionExplanation(explanation)
                        
- Change (_, newValue) => setActiveTab(newValue)

## System Responses

- Update ActiveTab
- Update SelectedRegionExplanation
- Update CurrentExplanationLevel
- Update SelectedRegionExplanation
- Update ShowSaliency

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `user_journey_analysispage.bpmn`.

## Metadata

- **Elements Count**: 12
- **Workflow Type**: user_journey
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
