# User Journey - AnnotationTools

**Workflow ID**: `user_journey_annotationtools`
**Type**: User Journey
**Last Updated**: Unknown

## Overview

User journey workflow for AnnotationTools component

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | User Starts Journey |  |
| user_task_1 | userTask | Click  => EditAnnotation(annotation) | actor=user |
| user_task_2 | userTask | Click  => DeleteAnnotation(annotation.id) | actor=user |
| user_task_3 | userTask | Click  => setExpanded(!expanded) | actor=user |
| user_task_4 | userTask | Click  => AddAnnotation | actor=user |
| user_task_5 | userTask | Click  => AddAnnotation(region.region_id) | actor=user |
| system_task_6 | serviceTask | Update Loading | actor=system |
| system_task_7 | serviceTask | Update Annotations | actor=system |
| system_task_8 | serviceTask | Update Error | actor=system |
| system_task_9 | serviceTask | Update Loading | actor=system |
| system_task_10 | serviceTask | Update NewAnnotation | actor=system |
| end_11 | endEvent | Journey Complete |  |

## User Actions

- Click  => EditAnnotation(annotation)
- Click  => DeleteAnnotation(annotation.id)
- Click  => setExpanded(!expanded)
- Click  => AddAnnotation
- Click  => AddAnnotation(region.region_id)

## System Responses

- Update Loading
- Update Annotations
- Update Error
- Update Loading
- Update NewAnnotation

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `user_journey_annotationtools.bpmn`.

## Metadata

- **Elements Count**: 12
- **Workflow Type**: user_journey
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
