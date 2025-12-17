# Children's Drawing Anomaly Detection System

**Version**: 0.1.0  
**Generated**: 2025-12-16 15:53:08

## Description
Machine learning system for detecting anomalies in children's drawings

## Base Information
- **OpenAPI Version**: 3.1.0
- **Total Endpoints**: 72
- **Documentation Format**: Enhanced with examples and specifications

## Quick Navigation
- [Authentication](./authentication.md) - Authentication methods and requirements
- [Error Handling](./error-handling.md) - Error response specifications
- [Request/Response Examples](./examples.json) - Comprehensive API examples
- [OpenAPI Schema](./openapi.json) - Complete OpenAPI specification
- [Endpoint Documentation](./endpoints/) - Detailed endpoint documentation

## Authentication Summary
No authentication required for this API.

## Error Handling Summary
Common error responses across 1 status codes:

- **422**: Used by 49 endpoint(s)


## Endpoint Categories

### Drawings
- [POST /api/v1/drawings/upload](./endpoints/POST__api_v1_drawings_upload.md)
- [GET /api/v1/drawings/upload/progress/{upload_id}](./endpoints/GET__api_v1_drawings_upload_progress_upload_id.md)
- [GET /api/v1/drawings/{drawing_id}](./endpoints/GET__api_v1_drawings_drawing_id.md)
- [DELETE /api/v1/drawings/{drawing_id}](./endpoints/DELETE__api_v1_drawings_drawing_id.md)
- [GET /api/v1/drawings/{drawing_id}/file](./endpoints/GET__api_v1_drawings_drawing_id_file.md)
- [GET /api/v1/drawings/](./endpoints/GET__api_v1_drawings_.md)
- [POST /api/v1/drawings/batch/upload](./endpoints/POST__api_v1_drawings_batch_upload.md)
- [GET /api/v1/drawings/stats](./endpoints/GET__api_v1_drawings_stats.md)

### Analysis
- [GET /api/v1/analysis/stats](./endpoints/GET__api_v1_analysis_stats.md)
- [POST /api/v1/analysis/analyze/{drawing_id}](./endpoints/POST__api_v1_analysis_analyze_drawing_id.md)
- [POST /api/v1/analysis/batch](./endpoints/POST__api_v1_analysis_batch.md)
- [GET /api/v1/analysis/batch/{batch_id}/progress](./endpoints/GET__api_v1_analysis_batch_batch_id_progress.md)
- [GET /api/v1/analysis/{analysis_id}](./endpoints/GET__api_v1_analysis_analysis_id.md)
- [POST /api/v1/analysis/embeddings/{drawing_id}](./endpoints/POST__api_v1_analysis_embeddings_drawing_id.md)
- [GET /api/v1/analysis/drawing/{drawing_id}](./endpoints/GET__api_v1_analysis_drawing_drawing_id.md)

### Interpretability
- [GET /api/v1/interpretability/{analysis_id}/interactive](./endpoints/GET__api_v1_interpretability_analysis_id_interactive.md) - Interactive saliency analysis
- [GET /api/v1/interpretability/{analysis_id}/simplified](./endpoints/GET__api_v1_interpretability_analysis_id_simplified.md) - Simplified explanations
- [GET /api/v1/interpretability/{analysis_id}/confidence](./endpoints/GET__api_v1_interpretability_analysis_id_confidence.md) - Confidence metrics
- [POST /api/v1/interpretability/{analysis_id}/export](./endpoints/POST__api_v1_interpretability_analysis_id_export.md) - Export results
- [GET /api/v1/interpretability/examples/{age_group}](./endpoints/GET__api_v1_interpretability_examples_age_group.md) - Educational examples

### Models
- [GET /api/v1/models/age-groups](./endpoints/GET__api_v1_models_age-groups.md)
- [POST /api/v1/models/train](./endpoints/POST__api_v1_models_train.md)
- [GET /api/v1/models/training/{job_id}/status](./endpoints/GET__api_v1_models_training_job_id_status.md)
- [PUT /api/v1/models/{model_id}/threshold](./endpoints/PUT__api_v1_models_model_id_threshold.md)
- [GET /api/v1/models/status](./endpoints/GET__api_v1_models_status.md)
- [POST /api/v1/models/auto-create](./endpoints/POST__api_v1_models_auto-create.md)
- [GET /api/v1/models/creation/{job_id}/status](./endpoints/GET__api_v1_models_creation_job_id_status.md)
- [DELETE /api/v1/models/{model_id}](./endpoints/DELETE__api_v1_models_model_id.md)
- [GET /api/v1/models/data-sufficiency/analyze](./endpoints/GET__api_v1_models_data-sufficiency_analyze.md)
- [GET /api/v1/models/data-sufficiency/age-group/{age_min}/{age_max}](./endpoints/GET__api_v1_models_data-sufficiency_age-group_age_min_age_max.md)
- [POST /api/v1/models/data-sufficiency/merge-age-groups](./endpoints/POST__api_v1_models_data-sufficiency_merge-age-groups.md)
- [GET /api/v1/models/data-sufficiency/warnings](./endpoints/GET__api_v1_models_data-sufficiency_warnings.md)

### Training
- [POST /api/v1/training/jobs](./endpoints/POST__api_v1_training_jobs.md)
- [GET /api/v1/training/jobs](./endpoints/GET__api_v1_training_jobs.md)
- [GET /api/v1/training/jobs/{job_id}](./endpoints/GET__api_v1_training_jobs_job_id.md)
- [POST /api/v1/training/jobs/{job_id}/cancel](./endpoints/POST__api_v1_training_jobs_job_id_cancel.md)
- [GET /api/v1/training/jobs/{job_id}/reports](./endpoints/GET__api_v1_training_jobs_job_id_reports.md)
- [POST /api/v1/training/deploy](./endpoints/POST__api_v1_training_deploy.md)
- [GET /api/v1/training/environments/status](./endpoints/GET__api_v1_training_environments_status.md)
- [POST /api/v1/training/sagemaker/setup](./endpoints/POST__api_v1_training_sagemaker_setup.md)
- [POST /api/v1/training/models/export](./endpoints/POST__api_v1_training_models_export.md)
- [GET /api/v1/training/models/exports](./endpoints/GET__api_v1_training_models_exports.md)
- [POST /api/v1/training/models/validate](./endpoints/POST__api_v1_training_models_validate.md)
- [POST /api/v1/training/models/deploy](./endpoints/POST__api_v1_training_models_deploy.md)
- [GET /api/v1/training/models/deployed](./endpoints/GET__api_v1_training_models_deployed.md)
- [POST /api/v1/training/models/{model_id}/undeploy](./endpoints/POST__api_v1_training_models_model_id_undeploy.md)

### Configuration
- [GET /api/v1/config/](./endpoints/GET__api_v1_config_.md)
- [PUT /api/v1/config/](./endpoints/PUT__api_v1_config_.md)
- [PUT /api/v1/config/threshold](./endpoints/PUT__api_v1_config_threshold.md)
- [PUT /api/v1/config/age-grouping](./endpoints/PUT__api_v1_config_age-grouping.md)
- [GET /api/v1/config/health](./endpoints/GET__api_v1_config_health.md)
- [GET /api/v1/config/stats](./endpoints/GET__api_v1_config_stats.md)
- [POST /api/v1/config/reset](./endpoints/POST__api_v1_config_reset.md)

### Health
- [GET /api/v1/health](./endpoints/GET__api_v1_health.md)
- [GET /api/v1/health/detailed](./endpoints/GET__api_v1_health_detailed.md)
- [GET /api/v1/health/component/{component_name}](./endpoints/GET__api_v1_health_component_component_name.md)
- [GET /api/v1/metrics](./endpoints/GET__api_v1_metrics.md)
- [GET /api/v1/metrics/history](./endpoints/GET__api_v1_metrics_history.md)
- [GET /api/v1/alerts](./endpoints/GET__api_v1_alerts.md)
- [POST /api/v1/alerts/thresholds](./endpoints/POST__api_v1_alerts_thresholds.md)
- [GET /api/v1/status](./endpoints/GET__api_v1_status.md)

### Backup
- [POST /api/v1/backup/full](./endpoints/POST__api_v1_backup_full.md)
- [POST /api/v1/backup/database](./endpoints/POST__api_v1_backup_database.md)
- [GET /api/v1/backup/list](./endpoints/GET__api_v1_backup_list.md)
- [GET /api/v1/backup/download/{backup_name}](./endpoints/GET__api_v1_backup_download_backup_name.md)
- [POST /api/v1/backup/restore](./endpoints/POST__api_v1_backup_restore.md)
- [POST /api/v1/backup/upload](./endpoints/POST__api_v1_backup_upload.md)
- [POST /api/v1/export](./endpoints/POST__api_v1_export.md)
- [GET /api/v1/export/download/{export_name}](./endpoints/GET__api_v1_export_download_export_name.md)
- [DELETE /api/v1/backup/{backup_name}](./endpoints/DELETE__api_v1_backup_backup_name.md)
- [POST /api/v1/cleanup](./endpoints/POST__api_v1_cleanup.md)
- [GET /api/v1/storage/info](./endpoints/GET__api_v1_storage_info.md)
- [POST /api/v1/storage/cleanup](./endpoints/POST__api_v1_storage_cleanup.md)

### Untagged
- [GET /](./endpoints/GET__.md)
- [GET /health](./endpoints/GET__health.md)
- [GET /health/detailed](./endpoints/GET__health_detailed.md)
- [GET /metrics](./endpoints/GET__metrics.md)


## Interactive Documentation
- **Swagger UI**: Available at `/docs` when running the API server
- **ReDoc**: Available at `/redoc` when running the API server

## Development
This documentation is automatically generated from the OpenAPI specification.
To regenerate, run:
```bash
python scripts/generate_docs.py --category api
```
