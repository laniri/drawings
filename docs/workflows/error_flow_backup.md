# Error Flow - Backup

**Workflow ID**: `error_flow_backup`
**Type**: Error Flow
**Last Updated**: Unknown

## Overview

Error handling workflow for backup

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | Error Condition Detected |  |
| error_event_1 | errorEvent | StorageError Error | error_type=StorageError |
| recovery_task_2 | task | Handle StorageError | recovery=True |
| error_event_3 | errorEvent | StorageError Error | error_type=StorageError |
| recovery_task_4 | task | Handle StorageError | recovery=True |
| error_event_5 | errorEvent | StorageError Error | error_type=StorageError |
| recovery_task_6 | task | Handle StorageError | recovery=True |
| end_7 | endEvent | Error Resolved |  |

## Error Conditions

- StorageError
- StorageError
- StorageError
- HTTPException
- StorageError
- HTTPException
- ConfigurationError
- HTTPException
- HTTPException
- Exception
- Exception
- Exception
- HTTPException
- HTTPException
- HTTPException
- HTTPException
- HTTPException
- HTTPException
- HTTPException
- except
- HTTPException
- HTTPException
- HTTPException
- HTTPException
- HTTPException
- HTTPException
- except
- HTTPException
- HTTPException
- HTTPException
- HTTPException
- HTTPException
- except
- HTTPException
- HTTPException
- except
- HTTPException
- HTTPException
- HTTPException
- HTTPException

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `error_flow_backup.bpmn`.

## Metadata

- **Elements Count**: 8
- **Workflow Type**: error_flow
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
