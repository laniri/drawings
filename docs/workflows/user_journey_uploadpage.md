# User Journey - UploadPage

**Workflow ID**: `user_journey_uploadpage`
**Type**: User Journey
**Last Updated**: Unknown

## Overview

User journey workflow for UploadPage component

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | User Starts Journey |  |
| user_task_1 | userTask | Click removeFile | actor=user |
| user_task_2 | userTask | Click  => navigate(`/analysis/${analysisId | actor=user |
| user_task_3 | userTask | Submit Submit(onSubmit) | actor=user |
| user_task_4 | userTask | Change (e) => field.onChange(parseFloat(e.target.value)) | actor=user |
| system_task_5 | serviceTask | POST Request to /api/drawings/upload | actor=system |
| system_task_6 | serviceTask | Update UploadProgress | actor=system |
| system_task_7 | serviceTask | Update UploadError | actor=system |
| system_task_8 | serviceTask | Update UploadedFile | actor=system |
| system_task_9 | serviceTask | Update UploadProgress | actor=system |
| end_10 | endEvent | Journey Complete |  |

## User Actions

- Click removeFile
- Click  => navigate(`/analysis/${analysisId
- Submit Submit(onSubmit)
- Change (e) => field.onChange(parseFloat(e.target.value))

## System Responses

- POST Request to /api/drawings/upload
- Update UploadProgress
- Update UploadError
- Update UploadedFile
- Update UploadProgress

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `user_journey_uploadpage.bpmn`.

## Metadata

- **Elements Count**: 11
- **Workflow Type**: user_journey
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
