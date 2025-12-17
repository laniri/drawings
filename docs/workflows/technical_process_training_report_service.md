# Technical Process - Training Report Service

**Workflow ID**: `technical_process_training_report_service`
**Type**: Technical Process
**Last Updated**: Unknown

## Overview

Technical process workflow for training_report_service service

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | Process Start |  |
| service_task_1 | serviceTask | Get Training Report Generator | service=training_report_service, method=get_training_report_generator |
| service_task_2 | serviceTask | To Dict | service=training_report_service, method=to_dict |
| service_task_3 | serviceTask | To Dict | service=training_report_service, method=to_dict |
| service_task_4 | serviceTask | To Dict | service=training_report_service, method=to_dict |
| service_task_5 | serviceTask |   Init   | service=training_report_service, method=__init__ |
| end_6 | endEvent | Process Complete |  |

## Process Steps

- mkdir
- use
- info
- info
- mkdir
- _calculate_comprehensive_metrics
- _extract_architecture_info
- _extract_configuration_info
- _generate_all_visualizations
- _generate_summary_statistics
- _analyze_model_performance
- _generate_training_recommendations
- _save_report_files
- info
- to_dict
- to_dict
- to_dict
- _update_database_record
- get
- error
- isoformat
- strftime
- now
- now
- get
- get
- mean
- std
- mean
- std
- _detect_convergence
- _detect_overfitting
- get
- get
- index
- get
- get
- get
- get
- get
- get
- get
- var
- var
- error
- mean
- mean
- polyfit
- polyfit
- get
- update
- get
- get
- get
- get
- get
- get
- get
- get
- get
- get
- get
- get
- get
- get
- get
- get
- get
- get
- get
- get
- get
- get
- get
- _create_loss_curves_plot
- _create_metrics_dashboard
- _create_error_distribution_plot
- _create_training_progress_plot
- _create_architecture_diagram
- _create_performance_summary_plot
- error
- get
- subplots
- plot
- plot
- set_xlabel
- set_ylabel
- set_title
- legend
- grid
- semilogy
- semilogy
- set_xlabel
- set_ylabel
- set_title
- legend
- grid
- tight_layout
- savefig
- close
- subplots
- bar
- set_title
- set_ylabel
- arange
- bar
- bar
- set_title
- set_ylabel
- set_xticks
- set_xticklabels
- legend
- bar
- set_title
- set_ylabel
- tick_params
- bar
- set_title
- set_ylabel
- bar
- set_title
- set_ylabel
- set_ylim
- text
- set_title
- axis
- tight_layout
- savefig
- close
- get
- keys
- values
- keys
- values
- get
- get
- get
- seed
- normal
- clip
- subplots
- hist
- axvline
- axvline
- set_xlabel
- set_ylabel
- set_title
- legend
- grid
- boxplot
- set_ylabel
- set_title
- grid
- tight_layout
- savefig
- close
- get
- get
- subplots
- plot
- plot
- scatter
- set_xlabel
- set_ylabel
- set_title
- legend
- grid
- tight_layout
- savefig
- close
- index
- axvspan
- axvspan
- get
- subplots
- get
- get
- get
- linspace
- set_title
- set_xlim
- set_ylim
- axis
- text
- tight_layout
- savefig
- close
- Rectangle
- add_patch
- text
- arrow
- get
- subplots
- tolist
- plot
- fill
- set_xticks
- set_xticklabels
- set_ylim
- set_title
- grid
- bar
- set_title
- set_ylabel
- tick_params
- barh
- set_title
- set_xlabel
- set_xlim
- bar
- set_title
- set_ylabel
- tick_params
- tight_layout
- savefig
- close
- keys
- values
- keys
- values
- linspace
- values
- keys
- values
- _calculate_overall_score
- _calculate_overall_score
- append
- append
- append
- append
- append
- append
- append
- append
- get
- append
- append
- append
- append
- append
- append
- append
- append
- append
- _save_metrics_csv
- dump
- write
- _generate_text_summary
- items
- DataFrame
- to_csv
- items
- first
- commit
- info
- dumps
- get
- add
- error
- rollback
- filter
- dumps
- get
- query

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `technical_process_training_report_service.bpmn`.

## Metadata

- **Elements Count**: 7
- **Workflow Type**: technical_process
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
