# Error Flow - Dataset Preparation

**Workflow ID**: `error_flow_dataset_preparation`
**Type**: Error Flow
**Last Updated**: Unknown

## Overview

Error handling workflow for dataset_preparation

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | Error Condition Detected |  |
| error_event_1 | errorEvent | Exception Error | error_type=Exception |
| recovery_task_2 | task | Handle Exception | recovery=True |
| error_event_3 | errorEvent | Exception Error | error_type=Exception |
| recovery_task_4 | task | Handle Exception | recovery=True |
| error_event_5 | errorEvent | json Error | error_type=json |
| recovery_task_6 | task | Handle json | recovery=True |
| end_7 | endEvent | Error Resolved |  |

## Error Conditions

- Exception
- Exception
- json
- Exception
- Exception
- ValueError
- ValueError
- ValueError
- FileNotFoundError
- FileNotFoundError
- ValidationError
- ValidationError
- ValidationError
- ValidationError
- ValidationError
- ValidationError
- ValidationError
- ValidationError
- ValidationError
- ValidationError
- ValidationError
- ValidationError
- ValidationError
- ValidationError

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `error_flow_dataset_preparation.bpmn`.

## Metadata

- **Elements Count**: 8
- **Workflow Type**: error_flow
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
