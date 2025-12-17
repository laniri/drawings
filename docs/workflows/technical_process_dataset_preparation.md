# Technical Process - Dataset Preparation

**Workflow ID**: `technical_process_dataset_preparation`
**Type**: Technical Process
**Last Updated**: Unknown

## Overview

Technical process workflow for dataset_preparation service

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | Process Start |  |
| service_task_1 | serviceTask | Get Dataset Preparation Service | service=dataset_preparation, method=get_dataset_preparation_service |
| service_task_2 | serviceTask | Train Count | service=dataset_preparation, method=train_count |
| service_task_3 | serviceTask | Validation Count | service=dataset_preparation, method=validation_count |
| service_task_4 | serviceTask | Test Count | service=dataset_preparation, method=test_count |
| service_task_5 | serviceTask | Total Count | service=dataset_preparation, method=total_count |
| end_6 | endEvent | Process Complete |  |

## Process Steps

- info
- info
- _load_metadata_file
- info
- _match_files_with_metadata
- _validate_dataset
- info
- exists
- exists
- extend
- extend
- glob
- glob
- upper
- lower
- _load_csv_metadata
- error
- lower
- _load_json_metadata
- read_csv
- iterrows
- issubset
- get
- get
- get
- get
- items
- notna
- load
- pop
- append
- append
- warning
- keys
- warning
- warning
- info
- warning
- warning
- info
- arange
- arange
- digitize
- unique
- info
- warning
- array
- array
- error
- warning
- warning
- load_dataset_from_folder
- create_dataset_splits
- append
- append
- append
- append
- mean

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `technical_process_dataset_preparation.bpmn`.

## Metadata

- **Elements Count**: 7
- **Workflow Type**: technical_process
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
