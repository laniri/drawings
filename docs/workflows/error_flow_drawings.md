# Error Flow - Drawings

**Workflow ID**: `error_flow_drawings`
**Type**: Error Flow
**Last Updated**: Unknown

## Overview

Error handling workflow for drawings

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | Error Condition Detected |  |
| error_event_1 | errorEvent | ValueError Error | error_type=ValueError |
| recovery_task_2 | task | Handle ValueError | recovery=True |
| error_event_3 | errorEvent | Exception Error | error_type=Exception |
| recovery_task_4 | task | Handle Exception | recovery=True |
| error_event_5 | errorEvent | Exception Error | error_type=Exception |
| recovery_task_6 | task | Handle Exception | recovery=True |
| end_7 | endEvent | Error Resolved |  |

## Error Conditions

- ValueError
- Exception
- Exception
- Exception
- Exception
- Exception
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
- HTTPException
- HTTPException
- HTTPException
- HTTPException

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `error_flow_drawings.bpmn`.

## Metadata

- **Elements Count**: 8
- **Workflow Type**: error_flow
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
