# Error Flow - Middleware

**Workflow ID**: `error_flow_middleware`
**Type**: Error Flow
**Last Updated**: Unknown

## Overview

Error handling workflow for middleware

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | Error Condition Detected |  |
| error_event_1 | errorEvent | DrawingAnalysisException Error | error_type=DrawingAnalysisException |
| recovery_task_2 | task | Handle DrawingAnalysisException | recovery=True |
| error_event_3 | errorEvent | DrawingAnalysisException Error | error_type=DrawingAnalysisException |
| recovery_task_4 | task | Handle DrawingAnalysisException | recovery=True |
| end_5 | endEvent | Error Resolved |  |

## Error Conditions

- DrawingAnalysisException
- DrawingAnalysisException
- except
- DrawingAnalysisException

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `error_flow_middleware.bpmn`.

## Metadata

- **Elements Count**: 6
- **Workflow Type**: error_flow
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
