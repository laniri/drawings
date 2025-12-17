# Technical Process - Local Training Environment

**Workflow ID**: `technical_process_local_training_environment`
**Type**: Technical Process
**Last Updated**: Unknown

## Overview

Technical process workflow for local_training_environment service

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | Process Start |  |
| service_task_1 | serviceTask | Get Local Training Environment | service=local_training_environment, method=get_local_training_environment |
| service_task_2 | serviceTask | Epoch Progress | service=local_training_environment, method=epoch_progress |
| service_task_3 | serviceTask | Batch Progress | service=local_training_environment, method=batch_progress |
| service_task_4 | serviceTask |   Init   | service=local_training_environment, method=__init__ |
| service_task_5 | serviceTask |  Detect Device | service=local_training_environment, method=_detect_device |
| end_6 | endEvent | Process Complete |  |

## Process Steps

- _detect_device
- is_available
- info
- _setup_cuda_device
- error
- is_available
- _setup_mps_device
- _setup_cpu_device
- device
- get_device_properties
- set_per_process_memory_fraction
- info
- device_count
- device
- info
- device
- cpu_count
- cpu_count
- virtual_memory
- set_num_threads
- info
- copy
- warning
- virtual_memory
- memory_allocated
- memory_reserved
- max_memory_allocated
- empty_cache
- info
- info
- set_num_threads
- info
- time
- Queue
- append
- time
- info
- get_memory_usage
- put
- time
- time
- warning
- append
- info
- time
- warning
- get_nowait
- copy
- info
- is_ready
- initialize
- get_memory_usage
- get_service_info
- is_available
- keys
- is_available
- info
- prepare_dataset
- info
- _generate_embeddings_for_split
- _generate_embeddings_for_split
- info
- _generate_embeddings_for_split
- array
- error
- array
- convert
- generate_embedding
- append
- warning
- open
- add
- commit
- refresh
- info
- Thread
- start
- error
- dumps
- utcnow
- first
- commit
- info
- add_callback
- optimize_for_training
- train
- _save_training_results
- utcnow
- commit
- info
- clear_cache
- error
- stop
- filter
- first
- utcnow
- commit
- error
- query
- filter
- query
- info
- mkdir
- add
- commit
- info
- dump
- _generate_training_plots
- error
- get
- dumps
- get
- figure
- plot
- plot
- xlabel
- ylabel
- title
- legend
- grid
- savefig
- close
- get
- info
- subplots
- plot
- plot
- set_title
- set_xlabel
- set_ylabel
- legend
- get
- tight_layout
- savefig
- close
- warning
- bar
- set_title
- set_ylabel
- bar
- set_title
- set_ylabel
- text
- set_title
- axis
- get
- get
- get
- get
- get
- get
- first
- get_latest_progress
- filter
- query
- all
- get_job_status
- order_by
- desc
- filter
- query
- stop
- first
- info
- utcnow
- commit
- filter
- query
- __init__
- time
- info
- to
- Adam
- _create_data_loader
- _create_data_loader
- _calculate_enhanced_metrics
- info
- parameters
- start_epoch
- _train_epoch
- _validate_epoch
- append
- update_epoch
- load_state_dict
- time
- get_architecture_info
- error
- copy
- info
- state_dict
- FloatTensor
- train
- to
- to
- zero_grad
- model
- criterion
- backward
- step
- item
- update_batch
- item
- eval
- no_grad
- to
- to
- model
- criterion
- item
- eval
- _calculate_set_metrics
- _calculate_set_metrics
- to
- no_grad
- model
- mean
- item
- item
- item
- item
- sort
- FloatTensor
- item
- mean
- std
- min
- max

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `technical_process_local_training_environment.bpmn`.

## Metadata

- **Elements Count**: 7
- **Workflow Type**: technical_process
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
