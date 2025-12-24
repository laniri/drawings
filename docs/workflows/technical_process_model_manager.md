# Technical Process - Model Manager

**Workflow ID**: `technical_process_model_manager`
**Type**: Technical Process
**Last Updated**: Unknown

## Overview

Technical process workflow for model_manager service

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | Process Start |  |
| service_task_1 | serviceTask | Get Model Manager | service=model_manager, method=get_model_manager |
| service_task_2 | serviceTask |   Init   | service=model_manager, method=__init__ |
| service_task_3 | serviceTask | Forward | service=model_manager, method=forward |
| service_task_4 | serviceTask | Encode | service=model_manager, method=encode |
| service_task_5 | serviceTask | Decode | service=model_manager, method=decode |
| end_6 | endEvent | Process Complete |  |

## Process Steps

- __init__
- Sequential
- Sequential
- extend
- append
- Linear
- extend
- Linear
- ReLU
- Dropout
- ReLU
- Dropout
- encoder
- decoder
- encoder
- decoder
- numel
- numel
- parameters
- parameters
- append
- _get_device
- MSELoss
- is_available
- device
- device
- is_available
- device
- device
- FloatTensor
- FloatTensor
- info
- to
- Adam
- ReduceLROnPlateau
- _prepare_data
- _calculate_metrics
- info
- get_architecture_info
- parameters
- time
- train
- eval
- append
- step
- load_state_dict
- get_architecture_info
- error
- to
- to
- zero_grad
- model
- criterion
- backward
- clip_grad_norm_
- step
- item
- isfinite
- error
- no_grad
- time
- warning
- copy
- info
- info
- error
- parameters
- tensor
- to
- to
- model
- criterion
- item
- join
- state_dict
- eval
- to
- no_grad
- model
- item
- mean
- item
- item
- item
- item
- sort
- item
- item
- FloatTensor
- criterion
- mean
- std
- min
- max
- mkdir
- all
- info
- first
- array
- filter
- append
- append
- order_by
- warning
- query
- desc
- filter
- query
- info
- _get_embeddings_for_age_group
- info
- info
- train
- add
- commit
- refresh
- _get_model_path
- save
- info
- error
- get
- values
- values
- warning
- dumps
- dumps
- state_dict
- get_architecture_info
- train_subject_aware_age_group_model
- first
- _get_model_path
- load
- load_state_dict
- eval
- warning
- warning
- exists
- warning
- info
- info
- error
- filter
- query
- get
- load_model
- unsqueeze
- no_grad
- item
- error
- FloatTensor
- mean
- load_model
- unsqueeze
- no_grad
- item
- item
- item
- error
- FloatTensor
- mean
- mean
- mean
- first
- loads
- get
- get
- filter
- loads
- warning
- query
- all
- get_model_info
- order_by
- query
- mean
- predict
- array
- _calculate_reconstruction_loss
- append
- mean
- std
- min
- max
- compute_subject_aware_reconstruction_loss
- error
- compute_anomaly_score
- first
- error
- filter
- query
- all
- error
- filter
- compute_anomaly_score
- warning
- query
- all
- info
- append
- append
- append
- error
- query
- append
- clear
- info

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `technical_process_model_manager.bpmn`.

## Metadata

- **Elements Count**: 7
- **Workflow Type**: technical_process
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
