# Technical Process - Sagemaker Training Service

**Workflow ID**: `technical_process_sagemaker_training_service`
**Type**: Technical Process
**Last Updated**: Unknown

## Overview

Technical process workflow for sagemaker_training_service service

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | Process Start |  |
| service_task_1 | serviceTask | Get Sagemaker Training Service | service=sagemaker_training_service, method=get_sagemaker_training_service |
| service_task_2 | serviceTask | To Sagemaker Config | service=sagemaker_training_service, method=to_sagemaker_config |
| service_task_3 | serviceTask |   Init   | service=sagemaker_training_service, method=__init__ |
| service_task_4 | serviceTask |  Initialize Docker | service=sagemaker_training_service, method=_initialize_docker |
| service_task_5 | serviceTask | Build Training Container | service=sagemaker_training_service, method=build_training_container |
| end_6 | endEvent | Process Complete |  |

## Process Steps

- warning
- info
- _initialize_docker
- from_env
- ping
- info
- error
- info
- _generate_dockerfile
- _generate_training_script
- _generate_requirements
- TemporaryDirectory
- write_text
- write_text
- write_text
- _copy_source_files
- build
- info
- error
- debug
- strip
- info
- get
- tag
- push
- info
- error
- debug
- _initialize_aws_clients
- client
- client
- client
- list_training_jobs
- info
- error
- error
- list_training_jobs
- list_buckets
- get_user
- append
- append
- append
- append
- get
- append
- create_role
- info
- attach_role_policy
- dumps
- get_role
- error
- info
- upload_file
- iterdir
- info
- error
- lower
- upload_file
- strftime
- to_sagemaker_config
- create_training_job
- add
- commit
- refresh
- info
- Thread
- start
- join
- utcnow
- error
- utcnow
- dumps
- utcnow
- info
- describe_training_job
- first
- sleep
- lower
- commit
- error
- error
- filter
- utcnow
- commit
- _process_completed_job
- query
- get
- get
- add
- commit
- info
- _download_training_artifacts
- exists
- error
- get
- load
- get
- get
- dumps
- get
- split
- mkdir
- list_objects_v2
- error
- replace
- download_file
- info
- first
- filter
- describe_training_job
- update
- warning
- query
- get
- get
- get
- get
- get
- get
- get
- stop_training_job
- first
- info
- utcnow
- commit
- error
- filter
- query
- all
- order_by
- desc
- filter
- query

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `technical_process_sagemaker_training_service.bpmn`.

## Metadata

- **Elements Count**: 7
- **Workflow Type**: technical_process
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
