# User Journey - InteractiveInterpretabilityViewer

**Workflow ID**: `user_journey_interactiveinterpretabilityviewer`
**Type**: User Journey
**Last Updated**: Unknown

## Overview

User journey workflow for InteractiveInterpretabilityViewer component

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | User Starts Journey |  |
| user_task_1 | userTask | Click  => setShowAttentionPatches(!showAttentionPatches) | actor=user |
| user_task_2 | userTask | Click ZoomOut | actor=user |
| user_task_3 | userTask | Click ZoomIn | actor=user |
| user_task_4 | userTask | Click (e) => {
                      e.stopPropagation
                      RegionClick(region)
                     | actor=user |
| user_task_5 | userTask | Click  => setSelectedRegion(null) | actor=user |
| system_task_6 | serviceTask | Update HoveredRegion | actor=system |
| system_task_7 | serviceTask | Update SelectedRegion | actor=system |
| system_task_8 | serviceTask | Update ZoomLevel | actor=system |
| system_task_9 | serviceTask | Update ZoomLevel | actor=system |
| system_task_10 | serviceTask | Update IsDragging | actor=system |
| end_11 | endEvent | Journey Complete |  |

## User Actions

- Click  => setShowAttentionPatches(!showAttentionPatches)
- Click ZoomOut
- Click ZoomIn
- Click (e) => {
                      e.stopPropagation
                      RegionClick(region)
                    
- Click  => setSelectedRegion(null)

## System Responses

- Update HoveredRegion
- Update SelectedRegion
- Update ZoomLevel
- Update ZoomLevel
- Update IsDragging

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `user_journey_interactiveinterpretabilityviewer.bpmn`.

## Metadata

- **Elements Count**: 12
- **Workflow Type**: user_journey
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
