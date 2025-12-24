# User Journey - InterpretabilityEducationHub

**Workflow ID**: `user_journey_interpretabilityeducationhub`
**Type**: User Journey
**Last Updated**: Unknown

## Overview

User journey workflow for InterpretabilityEducationHub component

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | User Starts Journey |  |
| user_task_1 | userTask | Click  => setShowTutorial(true) | actor=user |
| user_task_2 | userTask | Click  => toggleSection('quickstart') | actor=user |
| user_task_3 | userTask | Click  => setActiveTab(1) | actor=user |
| user_task_4 | userTask | Click  => toggleSection('analysis') | actor=user |
| user_task_5 | userTask | Click  => toggleSection('concepts') | actor=user |
| system_task_6 | serviceTask | Update IsFirstVisit | actor=system |
| system_task_7 | serviceTask | Update ShowTutorial | actor=system |
| system_task_8 | serviceTask | Update Item | actor=system |
| system_task_9 | serviceTask | Update IsFirstVisit | actor=system |
| system_task_10 | serviceTask | Update ShowTutorial | actor=system |
| end_11 | endEvent | Journey Complete |  |

## User Actions

- Click  => setShowTutorial(true)
- Click  => toggleSection('quickstart')
- Click  => setActiveTab(1)
- Click  => toggleSection('analysis')
- Click  => toggleSection('concepts')

## System Responses

- Update IsFirstVisit
- Update ShowTutorial
- Update Item
- Update IsFirstVisit
- Update ShowTutorial

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `user_journey_interpretabilityeducationhub.bpmn`.

## Metadata

- **Elements Count**: 12
- **Workflow Type**: user_journey
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
