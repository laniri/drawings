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
| user_task_3 | userTask | Change (e) => setSubjectFilter(e.target.value) | actor=user |
| user_task_4 | userTask | Change (_, newValue) => setChartTab(newValue) | actor=user |
| system_task_5 | serviceTask | Update SubjectFilter | actor=system |
| system_task_6 | serviceTask | Update ChartTab | actor=system |
| end_7 | endEvent | Journey Complete |  |

## User Actions

- Click  => navigate('/upload')
- Click  => navigate(`/analysis/${analysis.id
- Change (e) => setSubjectFilter(e.target.value)
- Change (_, newValue) => setChartTab(newValue)

## System Responses

- Update SubjectFilter
- Update ChartTab

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `user_journey_dashboardpage.bpmn`.

## Metadata

- **Elements Count**: 8
- **Workflow Type**: user_journey
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
