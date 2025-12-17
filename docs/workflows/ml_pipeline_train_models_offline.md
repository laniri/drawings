# ML Pipeline - Train Models Offline

**Workflow ID**: `ml_pipeline_train_models_offline`
**Type**: Ml Pipeline
**Last Updated**: Unknown

## Overview

Machine learning pipeline workflow for train_models_offline

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | ML Pipeline Start |  |
| ml_task_1 | serviceTask | Model Training | pipeline=train_models_offline, step_type=ml |
| ml_task_2 | serviceTask | Model Saving | pipeline=train_models_offline, step_type=ml |
| end_3 | endEvent | ML Pipeline Complete |  |

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `ml_pipeline_train_models_offline.bpmn`.

## Metadata

- **Elements Count**: 4
- **Workflow Type**: ml_pipeline
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
