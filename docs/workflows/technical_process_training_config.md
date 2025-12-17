# Technical Process - Training Config

**Workflow ID**: `technical_process_training_config`
**Type**: Technical Process
**Last Updated**: Unknown

## Overview

Technical process workflow for training_config service

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | Process Start |  |
| service_task_1 | serviceTask | Get Training Config Manager | service=training_config, method=get_training_config_manager |
| service_task_2 | serviceTask |   Post Init   | service=training_config, method=__post_init__ |
| service_task_3 | serviceTask |  Validate Config | service=training_config, method=_validate_config |
| service_task_4 | serviceTask |   Init   | service=training_config, method=__init__ |
| service_task_5 | serviceTask | Create Default Configs | service=training_config, method=create_default_configs |
| end_6 | endEvent | Process Complete |  |

## Process Steps

- _validate_config
- exists
- exists
- mkdir
- mkdir
- mkdir
- mkdir
- mkdir
- info
- _save_yaml_config
- _save_yaml_config
- _save_yaml_config
- _save_yaml_config
- _save_yaml_config
- info
- dump
- exists
- _dict_to_config
- error
- lower
- lower
- safe_load
- lower
- load
- _config_to_dict
- info
- error
- lower
- lower
- dump
- lower
- dump
- copy
- copy
- items
- items
- _validate_config
- append
- append
- append
- append
- append
- append
- info
- product
- sample
- _config_to_dict
- _dict_to_config
- append
- _config_to_dict
- create

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `technical_process_training_config.bpmn`.

## Metadata

- **Elements Count**: 7
- **Workflow Type**: technical_process
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
