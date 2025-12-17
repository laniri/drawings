# Technical Process - Embedding Service

**Workflow ID**: `technical_process_embedding_service`
**Type**: Technical Process
**Last Updated**: Unknown

## Overview

Technical process workflow for embedding_service service

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | Process Start |  |
| service_task_1 | serviceTask | Get Embedding Service | service=embedding_service, method=get_embedding_service |
| service_task_2 | serviceTask | Initialize Embedding Service | service=embedding_service, method=initialize_embedding_service |
| service_task_3 | serviceTask | Get Embedding Pipeline | service=embedding_service, method=get_embedding_pipeline |
| service_task_4 | serviceTask |   Init   | service=embedding_service, method=__init__ |
| service_task_5 | serviceTask |  Detect Device | service=embedding_service, method=_detect_device |
| end_6 | endEvent | Process Complete |  |

## Process Steps

- initialize
- _detect_device
- is_available
- device
- info
- is_available
- get_device_name
- device_count
- device
- info
- device
- info
- get_device_properties
- copy
- memory_allocated
- memory_reserved
- max_memory_allocated
- mkdir
- hexdigest
- md5
- encode
- _get_model_hash
- _get_model_hash
- _get_cache_path
- info
- from_pretrained
- from_pretrained
- to
- eval
- info
- exists
- info
- info
- cpu
- to
- info
- load
- to
- info
- warning
- dump
- warning
- is_loaded
- info
- load_model
- info
- is_loaded
- is_ready
- get_model_info
- get_memory_usage
- is_ready
- resize
- processor
- to
- fromarray
- convert
- astype
- hexdigest
- md5
- tobytes
- numpy
- cpu
- keys
- is_ready
- _preprocess_image
- _generate_cache_key
- copy
- no_grad
- model
- numpy
- squeeze
- copy
- _manage_cache
- array
- concatenate
- cpu
- is_ready
- extend
- generate_embedding
- append
- info
- is_ready
- clear
- clear_cache
- info
- store_embedding
- retrieve_embedding
- invalidate_cache
- get_service_info
- get_storage_stats
- generate_embedding
- get_model_info
- error
- generate_batch_embeddings
- append
- error
- get_model_info
- append
- copy

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `technical_process_embedding_service.bpmn`.

## Metadata

- **Elements Count**: 7
- **Workflow Type**: technical_process
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
