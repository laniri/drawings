# Error Flow - Sagemaker Training Service

**Workflow ID**: `error_flow_sagemaker_training_service`
**Type**: Error Flow
**Last Updated**: Unknown

## Overview

Error handling workflow for sagemaker_training_service

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | Error Condition Detected |  |
| error_event_1 | errorEvent | DockerException Error | error_type=DockerException |
| recovery_task_2 | task | Handle DockerException | recovery=True |
| error_event_3 | errorEvent | DockerException Error | error_type=DockerException |
| recovery_task_4 | task | Handle DockerException | recovery=True |
| error_event_5 | errorEvent | Exception Error | error_type=Exception |
| recovery_task_6 | task | Handle Exception | recovery=True |
| end_7 | endEvent | Error Resolved |  |

## Error Conditions

- DockerException
- DockerException
- Exception
- Exception
- DockerException
- NoCredentialsError
- ClientError
- ClientError
- ClientError
- ClientError
- ClientError
- ClientError
- ClientError
- Exception
- ClientError
- ClientError
- ClientError
- Exception
- DockerContainerError
- DockerContainerError
- ValueError
- if
- DockerContainerError
- DockerContainerError
- SageMakerConfigurationError
- SageMakerConfigurationError
- SageMakerConfigurationError
- SageMakerError
- SageMakerJobError
- SageMakerJobError

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `error_flow_sagemaker_training_service.bpmn`.

## Metadata

- **Elements Count**: 8
- **Workflow Type**: error_flow
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
