# User Journey - DashboardPage

**Workflow ID**: `user_journey_dashboardpage`
**Type**: User Journey
**Last Updated**: Unknown

## Overview

User journey workflow for DashboardPage component

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | User Starts Journey |  |
| user_task_1 | userTask | Click  => navigate('/upload') | actor=user |
| user_task_2 | userTask | Click  => navigate(`/analysis/${analysis.id | actor=user |
| system_task_3 | serviceTask | GET Request to /api/analysis/stats | actor=system |
| end_4 | endEvent | Journey Complete |  |

## User Actions

- Click  => navigate('/upload')
- Click  => navigate(`/analysis/${analysis.id

## System Responses

- GET Request to /api/analysis/stats

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `user_journey_dashboardpage.bpmn`.

## Metadata

- **Elements Count**: 5
- **Workflow Type**: user_journey
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
