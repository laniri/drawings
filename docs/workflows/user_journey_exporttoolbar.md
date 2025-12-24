# User Journey - ExportToolbar

**Workflow ID**: `user_journey_exporttoolbar`
**Type**: User Journey
**Last Updated**: Unknown

## Overview

User journey workflow for ExportToolbar component

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | User Starts Journey |  |
| user_task_1 | userTask | Click MenuOpen | actor=user |
| user_task_2 | userTask | Click  => setHistoryDialogOpen(true) | actor=user |
| user_task_3 | userTask | Click  => QuickExport(format as ExportOptions['format']) | actor=user |
| user_task_4 | userTask | Click  => setExportDialogOpen(true) | actor=user |
| user_task_5 | userTask | Click  => setExportDialogOpen(false) | actor=user |
| system_task_6 | serviceTask | Update AnchorEl | actor=system |
| system_task_7 | serviceTask | Update AnchorEl | actor=system |
| system_task_8 | serviceTask | Update ExportDialogOpen | actor=system |
| system_task_9 | serviceTask | Update Exporting | actor=system |
| system_task_10 | serviceTask | Update ExportError | actor=system |
| end_11 | endEvent | Journey Complete |  |

## User Actions

- Click MenuOpen
- Click  => setHistoryDialogOpen(true)
- Click  => QuickExport(format as ExportOptions['format'])
- Click  => setExportDialogOpen(true)
- Click  => setExportDialogOpen(false)

## System Responses

- Update AnchorEl
- Update AnchorEl
- Update ExportDialogOpen
- Update Exporting
- Update ExportError

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `user_journey_exporttoolbar.bpmn`.

## Metadata

- **Elements Count**: 12
- **Workflow Type**: user_journey
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
