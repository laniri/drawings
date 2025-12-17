# Database Schema

This document describes the database schema for the Children's Drawing Anomaly Detection System.

## Table: drawings

**Purpose**: Data storage for drawings

### Columns

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | INTEGER | No | None | - |
| filename | VARCHAR | No | None | - |
| file_path | VARCHAR | No | None | - |
| age_years | FLOAT | No | None | - |
| subject | VARCHAR | Yes | None | - |
| expert_label | VARCHAR | Yes | None | - |
| drawing_tool | VARCHAR | Yes | None | - |
| prompt | TEXT | Yes | None | - |
| upload_timestamp | DATETIME | Yes | <function datetime.utcnow at 0x10caa4860> | - |

## Table: drawing_embeddings

**Purpose**: Data storage for drawing embeddings

### Columns

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | INTEGER | No | None | - |
| drawing_id | INTEGER | No | None | - |
| model_type | VARCHAR | No | None | - |
| embedding_vector | BLOB | No | None | - |
| vector_dimension | INTEGER | No | None | - |
| created_timestamp | DATETIME | Yes | <function datetime.utcnow at 0x10caa71a0> | - |

## Table: age_group_models

**Purpose**: Data storage for age group models

### Columns

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | INTEGER | No | None | - |
| age_min | FLOAT | No | None | - |
| age_max | FLOAT | No | None | - |
| model_type | VARCHAR | No | None | - |
| vision_model | VARCHAR | No | None | - |
| parameters | TEXT | No | None | - |
| sample_count | INTEGER | No | None | - |
| threshold | FLOAT | No | None | - |
| created_timestamp | DATETIME | Yes | <function datetime.utcnow at 0x10caa7c40> | - |
| is_active | BOOLEAN | Yes | True | - |

## Table: anomaly_analyses

**Purpose**: Data storage for anomaly analyses

### Columns

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | INTEGER | No | None | - |
| drawing_id | INTEGER | No | None | - |
| age_group_model_id | INTEGER | No | None | - |
| anomaly_score | FLOAT | No | None | - |
| normalized_score | FLOAT | No | None | - |
| is_anomaly | BOOLEAN | No | None | - |
| confidence | FLOAT | No | None | - |
| analysis_timestamp | DATETIME | Yes | <function datetime.utcnow at 0x10cad9080> | - |

## Table: interpretability_results

**Purpose**: Data storage for interpretability results

### Columns

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | INTEGER | No | None | - |
| analysis_id | INTEGER | No | None | - |
| saliency_map_path | VARCHAR | No | None | - |
| overlay_image_path | VARCHAR | No | None | - |
| explanation_text | TEXT | Yes | None | - |
| importance_regions | TEXT | Yes | None | - |
| created_timestamp | DATETIME | Yes | <function datetime.utcnow at 0x10cada480> | - |

## Table: training_jobs

**Purpose**: Data storage for training jobs

### Columns

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | INTEGER | No | None | - |
| job_name | VARCHAR | No | None | - |
| environment | VARCHAR | No | None | - |
| config_parameters | TEXT | No | None | - |
| dataset_path | VARCHAR | No | None | - |
| status | VARCHAR | No | None | - |
| start_timestamp | DATETIME | Yes | None | - |
| end_timestamp | DATETIME | Yes | None | - |
| sagemaker_job_arn | VARCHAR | Yes | None | - |

## Table: training_reports

**Purpose**: Data storage for training reports

### Columns

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | INTEGER | No | None | - |
| training_job_id | INTEGER | No | None | - |
| final_loss | FLOAT | No | None | - |
| validation_accuracy | FLOAT | No | None | - |
| best_epoch | INTEGER | No | None | - |
| training_time_seconds | FLOAT | No | None | - |
| model_parameters_path | VARCHAR | No | None | - |
| metrics_summary | TEXT | No | None | - |
| report_file_path | VARCHAR | No | None | - |
| created_timestamp | DATETIME | Yes | <function datetime.utcnow at 0x10cb00360> | - |

