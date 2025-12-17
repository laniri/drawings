# Error Flow - Score Normalizer

**Workflow ID**: `error_flow_score_normalizer`
**Type**: Error Flow
**Last Updated**: Unknown

## Overview

Error handling workflow for score_normalizer

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | Error Condition Detected |  |
| error_event_1 | errorEvent | Exception Error | error_type=Exception |
| recovery_task_2 | task | Handle Exception | recovery=True |
| error_event_3 | errorEvent | ImportError Error | error_type=ImportError |
| recovery_task_4 | task | Handle ImportError | recovery=True |
| error_event_5 | errorEvent | Exception Error | error_type=Exception |
| recovery_task_6 | task | Handle Exception | recovery=True |
| end_7 | endEvent | Error Resolved |  |

## Error Conditions

- Exception
- ImportError
- Exception
- Exception
- Exception
- ScoreNormalizationError
- ScoreNormalizationError
- ScoreNormalizationError
- ScoreNormalizationError
- ScoreNormalizationError
- ScoreNormalizationError
- ScoreNormalizationError
- ScoreNormalizationError
- ScoreNormalizationError

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `error_flow_score_normalizer.bpmn`.

## Metadata

- **Elements Count**: 8
- **Workflow Type**: error_flow
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
