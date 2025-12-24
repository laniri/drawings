# User Journey - ConfigurationPage

**Workflow ID**: `user_journey_configurationpage`
**Type**: User Journey
**Last Updated**: Unknown

## Overview

User journey workflow for ConfigurationPage component

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | User Starts Journey |  |
| user_task_1 | userTask | Click  => setTrainDialogOpen(true) | actor=user |
| user_task_2 | userTask | Click  => deleteModelMutation.mutate(model.id) | actor=user |
| user_task_3 | userTask | Click  => setTrainDialogOpen(false) | actor=user |
| user_task_4 | userTask | Click TrainModel | actor=user |
| user_task_5 | userTask | Submit Submit(onSubmit) | actor=user |
| system_task_6 | serviceTask | GET Request to /api/config/ | actor=system |
| system_task_7 | serviceTask | GET Request to /api/models/age-groups | actor=system |
| system_task_8 | serviceTask | GET Request to /api/config/subject-statistics | actor=system |
| system_task_9 | serviceTask | PUT Request to /api/config/threshold | actor=system |
| system_task_10 | serviceTask | PUT Request to /api/config/age-grouping | actor=system |
| end_11 | endEvent | Journey Complete |  |

## User Actions

- Click  => setTrainDialogOpen(true)
- Click  => deleteModelMutation.mutate(model.id)
- Click  => setTrainDialogOpen(false)
- Click TrainModel
- Submit Submit(onSubmit)

## System Responses

- GET Request to /api/config/
- GET Request to /api/models/age-groups
- GET Request to /api/config/subject-statistics
- PUT Request to /api/config/threshold
- PUT Request to /api/config/age-grouping

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `user_journey_configurationpage.bpmn`.

## Metadata

- **Elements Count**: 12
- **Workflow Type**: user_journey
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
