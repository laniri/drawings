# Children's Drawing Anomaly Detection System

**Version**: 0.1.0  
**Generated**: 2026-01-10 09:24:53

## Description
Machine learning system for detecting anomalies in children's drawings

## Base Information
- **OpenAPI Version**: 3.1.0
- **Total Endpoints**: 299
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

- **422**: Used by 182 endpoint(s)


## Endpoint Categories

### Untagged
- [GET /health](./endpoints/GET__health.md)
- [GET /health/simple](./endpoints/GET__health_simple.md)
- [GET /](./endpoints/GET__.md)
- [GET /api](./endpoints/GET__api.md)
- [GET /health/detailed](./endpoints/GET__health_detailed.md)
- [GET /metrics](./endpoints/GET__metrics.md)
- [GET /monitoring/logs](./endpoints/GET__monitoring_logs.md)
- [GET /monitoring/alerts](./endpoints/GET__monitoring_alerts.md)

### Drawings
- [POST /api/v1/drawings/upload](./endpoints/POST__api_v1_drawings_upload.md)
- [GET /api/v1/drawings/upload/progress/{upload_id}](./endpoints/GET__api_v1_drawings_upload_progress_upload_id.md)
- [GET /api/v1/drawings/{drawing_id}](./endpoints/GET__api_v1_drawings_drawing_id.md)
- [DELETE /api/v1/drawings/{drawing_id}](./endpoints/DELETE__api_v1_drawings_drawing_id.md)
- [GET /api/v1/drawings/{drawing_id}/file](./endpoints/GET__api_v1_drawings_drawing_id_file.md)
- [GET /api/v1/drawings/](./endpoints/GET__api_v1_drawings_.md)
- [POST /api/v1/drawings/batch/upload](./endpoints/POST__api_v1_drawings_batch_upload.md)
- [GET /api/v1/drawings/stats](./endpoints/GET__api_v1_drawings_stats.md)
- [POST /api/drawings/upload](./endpoints/POST__api_drawings_upload.md)
- [GET /api/drawings/upload/progress/{upload_id}](./endpoints/GET__api_drawings_upload_progress_upload_id.md)
- [GET /api/drawings/{drawing_id}](./endpoints/GET__api_drawings_drawing_id.md)
- [DELETE /api/drawings/{drawing_id}](./endpoints/DELETE__api_drawings_drawing_id.md)
- [GET /api/drawings/{drawing_id}/file](./endpoints/GET__api_drawings_drawing_id_file.md)
- [GET /api/drawings/](./endpoints/GET__api_drawings_.md)
- [POST /api/drawings/batch/upload](./endpoints/POST__api_drawings_batch_upload.md)
- [GET /api/drawings/stats](./endpoints/GET__api_drawings_stats.md)

### Analysis
- [GET /api/v1/analysis/stats](./endpoints/GET__api_v1_analysis_stats.md)
- [POST /api/v1/analysis/analyze/{drawing_id}](./endpoints/POST__api_v1_analysis_analyze_drawing_id.md)
- [POST /api/v1/analysis/batch](./endpoints/POST__api_v1_analysis_batch.md)
- [GET /api/v1/analysis/batch/{batch_id}/progress](./endpoints/GET__api_v1_analysis_batch_batch_id_progress.md)
- [GET /api/v1/analysis/{analysis_id}](./endpoints/GET__api_v1_analysis_analysis_id.md)
- [POST /api/v1/analysis/embeddings/{drawing_id}](./endpoints/POST__api_v1_analysis_embeddings_drawing_id.md)
- [GET /api/v1/analysis/drawing/{drawing_id}](./endpoints/GET__api_v1_analysis_drawing_drawing_id.md)
- [GET /api/analysis/stats](./endpoints/GET__api_analysis_stats.md)
- [POST /api/analysis/analyze/{drawing_id}](./endpoints/POST__api_analysis_analyze_drawing_id.md)
- [POST /api/analysis/batch](./endpoints/POST__api_analysis_batch.md)
- [GET /api/analysis/batch/{batch_id}/progress](./endpoints/GET__api_analysis_batch_batch_id_progress.md)
- [GET /api/analysis/{analysis_id}](./endpoints/GET__api_analysis_analysis_id.md)
- [POST /api/analysis/embeddings/{drawing_id}](./endpoints/POST__api_analysis_embeddings_drawing_id.md)
- [GET /api/analysis/drawing/{drawing_id}](./endpoints/GET__api_analysis_drawing_drawing_id.md)

### Interpretability
- [GET /api/v1/interpretability/{analysis_id}/interactive](./endpoints/GET__api_v1_interpretability_analysis_id_interactive.md)
- [GET /api/v1/interpretability/{analysis_id}/simplified](./endpoints/GET__api_v1_interpretability_analysis_id_simplified.md)
- [GET /api/v1/interpretability/{analysis_id}/confidence](./endpoints/GET__api_v1_interpretability_analysis_id_confidence.md)
- [POST /api/v1/interpretability/{analysis_id}/export](./endpoints/POST__api_v1_interpretability_analysis_id_export.md)
- [GET /api/v1/interpretability/examples](./endpoints/GET__api_v1_interpretability_examples.md)
- [GET /api/v1/interpretability/examples/{age_group}](./endpoints/GET__api_v1_interpretability_examples_age_group.md)
- [GET /api/v1/interpretability/{analysis_id}/attribution](./endpoints/GET__api_v1_interpretability_analysis_id_attribution.md)
- [POST /api/v1/interpretability/{analysis_id}/annotate](./endpoints/POST__api_v1_interpretability_analysis_id_annotate.md)
- [GET /api/interpretability/{analysis_id}/interactive](./endpoints/GET__api_interpretability_analysis_id_interactive.md)
- [GET /api/interpretability/{analysis_id}/simplified](./endpoints/GET__api_interpretability_analysis_id_simplified.md)
- [GET /api/interpretability/{analysis_id}/confidence](./endpoints/GET__api_interpretability_analysis_id_confidence.md)
- [POST /api/interpretability/{analysis_id}/export](./endpoints/POST__api_interpretability_analysis_id_export.md)
- [GET /api/interpretability/examples](./endpoints/GET__api_interpretability_examples.md)
- [GET /api/interpretability/examples/{age_group}](./endpoints/GET__api_interpretability_examples_age_group.md)
- [GET /api/interpretability/{analysis_id}/attribution](./endpoints/GET__api_interpretability_analysis_id_attribution.md)
- [POST /api/interpretability/{analysis_id}/annotate](./endpoints/POST__api_interpretability_analysis_id_annotate.md)

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
- [GET /api/models/age-groups](./endpoints/GET__api_models_age-groups.md)
- [POST /api/models/train](./endpoints/POST__api_models_train.md)
- [GET /api/models/training/{job_id}/status](./endpoints/GET__api_models_training_job_id_status.md)
- [PUT /api/models/{model_id}/threshold](./endpoints/PUT__api_models_model_id_threshold.md)
- [GET /api/models/status](./endpoints/GET__api_models_status.md)
- [POST /api/models/auto-create](./endpoints/POST__api_models_auto-create.md)
- [GET /api/models/creation/{job_id}/status](./endpoints/GET__api_models_creation_job_id_status.md)
- [DELETE /api/models/{model_id}](./endpoints/DELETE__api_models_model_id.md)
- [GET /api/models/data-sufficiency/analyze](./endpoints/GET__api_models_data-sufficiency_analyze.md)
- [GET /api/models/data-sufficiency/age-group/{age_min}/{age_max}](./endpoints/GET__api_models_data-sufficiency_age-group_age_min_age_max.md)
- [POST /api/models/data-sufficiency/merge-age-groups](./endpoints/POST__api_models_data-sufficiency_merge-age-groups.md)
- [GET /api/models/data-sufficiency/warnings](./endpoints/GET__api_models_data-sufficiency_warnings.md)

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
- [POST /api/training/jobs](./endpoints/POST__api_training_jobs.md)
- [GET /api/training/jobs](./endpoints/GET__api_training_jobs.md)
- [GET /api/training/jobs/{job_id}](./endpoints/GET__api_training_jobs_job_id.md)
- [POST /api/training/jobs/{job_id}/cancel](./endpoints/POST__api_training_jobs_job_id_cancel.md)
- [GET /api/training/jobs/{job_id}/reports](./endpoints/GET__api_training_jobs_job_id_reports.md)
- [POST /api/training/deploy](./endpoints/POST__api_training_deploy.md)
- [GET /api/training/environments/status](./endpoints/GET__api_training_environments_status.md)
- [POST /api/training/sagemaker/setup](./endpoints/POST__api_training_sagemaker_setup.md)
- [POST /api/training/models/export](./endpoints/POST__api_training_models_export.md)
- [GET /api/training/models/exports](./endpoints/GET__api_training_models_exports.md)
- [POST /api/training/models/validate](./endpoints/POST__api_training_models_validate.md)
- [POST /api/training/models/deploy](./endpoints/POST__api_training_models_deploy.md)
- [GET /api/training/models/deployed](./endpoints/GET__api_training_models_deployed.md)
- [POST /api/training/models/{model_id}/undeploy](./endpoints/POST__api_training_models_model_id_undeploy.md)

### Configuration
- [GET /api/v1/config/](./endpoints/GET__api_v1_config_.md)
- [PUT /api/v1/config/](./endpoints/PUT__api_v1_config_.md)
- [PUT /api/v1/config/threshold](./endpoints/PUT__api_v1_config_threshold.md)
- [PUT /api/v1/config/age-grouping](./endpoints/PUT__api_v1_config_age-grouping.md)
- [GET /api/v1/config/health](./endpoints/GET__api_v1_config_health.md)
- [GET /api/v1/config/stats](./endpoints/GET__api_v1_config_stats.md)
- [GET /api/v1/config/subjects](./endpoints/GET__api_v1_config_subjects.md)
- [GET /api/v1/config/subjects/statistics](./endpoints/GET__api_v1_config_subjects_statistics.md)
- [GET /api/v1/config/models/subject-aware](./endpoints/GET__api_v1_config_models_subject-aware.md)
- [POST /api/v1/config/reset](./endpoints/POST__api_v1_config_reset.md)
- [GET /api/config/](./endpoints/GET__api_config_.md)
- [PUT /api/config/](./endpoints/PUT__api_config_.md)
- [PUT /api/config/threshold](./endpoints/PUT__api_config_threshold.md)
- [PUT /api/config/age-grouping](./endpoints/PUT__api_config_age-grouping.md)
- [GET /api/config/health](./endpoints/GET__api_config_health.md)
- [GET /api/config/stats](./endpoints/GET__api_config_stats.md)
- [GET /api/config/subjects](./endpoints/GET__api_config_subjects.md)
- [GET /api/config/subjects/statistics](./endpoints/GET__api_config_subjects_statistics.md)
- [GET /api/config/models/subject-aware](./endpoints/GET__api_config_models_subject-aware.md)
- [POST /api/config/reset](./endpoints/POST__api_config_reset.md)

### Documentation
- [GET /api/v1/documentation/status](./endpoints/GET__api_v1_documentation_status.md)
- [GET /api/v1/documentation/metrics](./endpoints/GET__api_v1_documentation_metrics.md)
- [POST /api/v1/documentation/generate](./endpoints/POST__api_v1_documentation_generate.md)
- [POST /api/v1/documentation/generate/sync](./endpoints/POST__api_v1_documentation_generate_sync.md)
- [GET /api/v1/documentation/categories](./endpoints/GET__api_v1_documentation_categories.md)
- [GET /api/v1/documentation/files](./endpoints/GET__api_v1_documentation_files.md)
- [DELETE /api/v1/documentation/cache](./endpoints/DELETE__api_v1_documentation_cache.md)
- [GET /api/v1/documentation/validation](./endpoints/GET__api_v1_documentation_validation.md)
- [POST /api/v1/documentation/validate](./endpoints/POST__api_v1_documentation_validate.md)
- [GET /api/v1/documentation/preview/{category}](./endpoints/GET__api_v1_documentation_preview_category.md)
- [POST /api/v1/documentation/batch/generate](./endpoints/POST__api_v1_documentation_batch_generate.md)
- [POST /api/v1/documentation/batch/validate](./endpoints/POST__api_v1_documentation_batch_validate.md)
- [GET /api/v1/documentation/schedule](./endpoints/GET__api_v1_documentation_schedule.md)
- [POST /api/v1/documentation/schedule](./endpoints/POST__api_v1_documentation_schedule.md)
- [POST /api/v1/documentation/search](./endpoints/POST__api_v1_documentation_search.md)
- [GET /api/v1/documentation/search/suggestions](./endpoints/GET__api_v1_documentation_search_suggestions.md)
- [GET /api/v1/documentation/search/statistics](./endpoints/GET__api_v1_documentation_search_statistics.md)
- [POST /api/v1/documentation/search/index](./endpoints/POST__api_v1_documentation_search_index.md)
- [GET /api/v1/documentation/navigation/{document_id}](./endpoints/GET__api_v1_documentation_navigation_document_id.md)
- [GET /api/v1/documentation/navigation/sitemap](./endpoints/GET__api_v1_documentation_navigation_sitemap.md)
- [GET /api/v1/documentation/navigation/cross-references](./endpoints/GET__api_v1_documentation_navigation_cross-references.md)
- [POST /api/v1/documentation/navigation/rebuild](./endpoints/POST__api_v1_documentation_navigation_rebuild.md)
- [GET /api/documentation/status](./endpoints/GET__api_documentation_status.md)
- [GET /api/documentation/metrics](./endpoints/GET__api_documentation_metrics.md)
- [POST /api/documentation/generate](./endpoints/POST__api_documentation_generate.md)
- [POST /api/documentation/generate/sync](./endpoints/POST__api_documentation_generate_sync.md)
- [GET /api/documentation/categories](./endpoints/GET__api_documentation_categories.md)
- [GET /api/documentation/files](./endpoints/GET__api_documentation_files.md)
- [DELETE /api/documentation/cache](./endpoints/DELETE__api_documentation_cache.md)
- [GET /api/documentation/validation](./endpoints/GET__api_documentation_validation.md)
- [POST /api/documentation/validate](./endpoints/POST__api_documentation_validate.md)
- [GET /api/documentation/preview/{category}](./endpoints/GET__api_documentation_preview_category.md)
- [POST /api/documentation/batch/generate](./endpoints/POST__api_documentation_batch_generate.md)
- [POST /api/documentation/batch/validate](./endpoints/POST__api_documentation_batch_validate.md)
- [GET /api/documentation/schedule](./endpoints/GET__api_documentation_schedule.md)
- [POST /api/documentation/schedule](./endpoints/POST__api_documentation_schedule.md)
- [POST /api/documentation/search](./endpoints/POST__api_documentation_search.md)
- [GET /api/documentation/search/suggestions](./endpoints/GET__api_documentation_search_suggestions.md)
- [GET /api/documentation/search/statistics](./endpoints/GET__api_documentation_search_statistics.md)
- [POST /api/documentation/search/index](./endpoints/POST__api_documentation_search_index.md)
- [GET /api/documentation/navigation/{document_id}](./endpoints/GET__api_documentation_navigation_document_id.md)
- [GET /api/documentation/navigation/sitemap](./endpoints/GET__api_documentation_navigation_sitemap.md)
- [GET /api/documentation/navigation/cross-references](./endpoints/GET__api_documentation_navigation_cross-references.md)
- [POST /api/documentation/navigation/rebuild](./endpoints/POST__api_documentation_navigation_rebuild.md)

### Metrics
**Updated January 2026**: Usage metrics endpoint now returns nested data structure. See [detailed documentation](./endpoints/metrics-usage.md).

- [GET /api/v1/metrics/usage](./endpoints/GET__api_v1_metrics_usage.md) - **UPDATED**: Comprehensive usage metrics with nested structure
- [GET /api/v1/metrics/health](./endpoints/GET__api_v1_metrics_health.md) - System health metrics
- [GET /api/v1/metrics/sessions](./endpoints/GET__api_v1_metrics_sessions.md) - Session-specific metrics
- [GET /api/v1/metrics/performance](./endpoints/GET__api_v1_metrics_performance.md) - Detailed performance metrics
- [POST /api/v1/metrics/session/start](./endpoints/POST__api_v1_metrics_session_start.md) - Start user session
- [POST /api/v1/metrics/session/{session_id}/end](./endpoints/POST__api_v1_metrics_session_session_id_end.md) - End user session
- [GET /api/v1/metrics/cloudwatch/status](./endpoints/GET__api_v1_metrics_cloudwatch_status.md) - CloudWatch integration status
- [GET /api/metrics/usage](./endpoints/GET__api_metrics_usage.md)
- [GET /api/metrics/health](./endpoints/GET__api_metrics_health.md)
- [GET /api/metrics/sessions](./endpoints/GET__api_metrics_sessions.md)
- [GET /api/metrics/performance](./endpoints/GET__api_metrics_performance.md)
- [POST /api/metrics/session/start](./endpoints/POST__api_metrics_session_start.md)
- [POST /api/metrics/session/{session_id}/end](./endpoints/POST__api_metrics_session_session_id_end.md)
- [GET /api/metrics/cloudwatch/status](./endpoints/GET__api_metrics_cloudwatch_status.md)

### Demo
- [GET /api/v1/demo/](./endpoints/GET__api_v1_demo_.md)
- [GET /api/v1/demo/samples](./endpoints/GET__api_v1_demo_samples.md)
- [GET /api/v1/demo/samples/{sample_id}](./endpoints/GET__api_v1_demo_samples_sample_id.md)
- [GET /api/v1/demo/project-info](./endpoints/GET__api_v1_demo_project-info.md)
- [GET /api/v1/demo/disclaimer](./endpoints/GET__api_v1_demo_disclaimer.md)
- [GET /api/v1/demo/technical-links](./endpoints/GET__api_v1_demo_technical-links.md)
- [GET /api/v1/demo/statistics](./endpoints/GET__api_v1_demo_statistics.md)
- [GET /api/demo/](./endpoints/GET__api_demo_.md)
- [GET /api/demo/samples](./endpoints/GET__api_demo_samples.md)
- [GET /api/demo/samples/{sample_id}](./endpoints/GET__api_demo_samples_sample_id.md)
- [GET /api/demo/project-info](./endpoints/GET__api_demo_project-info.md)
- [GET /api/demo/disclaimer](./endpoints/GET__api_demo_disclaimer.md)
- [GET /api/demo/technical-links](./endpoints/GET__api_demo_technical-links.md)
- [GET /api/demo/statistics](./endpoints/GET__api_demo_statistics.md)
- [GET /demo/](./endpoints/GET__demo_.md)
- [GET /demo/samples](./endpoints/GET__demo_samples.md)
- [GET /demo/samples/{sample_id}](./endpoints/GET__demo_samples_sample_id.md)
- [GET /demo/project-info](./endpoints/GET__demo_project-info.md)
- [GET /demo/disclaimer](./endpoints/GET__demo_disclaimer.md)
- [GET /demo/technical-links](./endpoints/GET__demo_technical-links.md)
- [GET /demo/statistics](./endpoints/GET__demo_statistics.md)

### Files
- [GET /api/v1/files/s3/{file_path}](./endpoints/GET__api_v1_files_s3_file_path.md)
- [HEAD /api/v1/files/s3/{file_path}](./endpoints/HEAD__api_v1_files_s3_file_path.md)
- [GET /api/v1/files/markdown](./endpoints/GET__api_v1_files_markdown.md)
- [GET /api/files/s3/{file_path}](./endpoints/GET__api_files_s3_file_path.md)
- [HEAD /api/files/s3/{file_path}](./endpoints/HEAD__api_files_s3_file_path.md)
- [GET /api/files/markdown](./endpoints/GET__api_files_markdown.md)

### Database
- [POST /api/v1/database/backup](./endpoints/POST__api_v1_database_backup.md)
- [POST /api/v1/database/migrate](./endpoints/POST__api_v1_database_migrate.md)
- [GET /api/v1/database/migration-info](./endpoints/GET__api_v1_database_migration-info.md)
- [POST /api/v1/database/validate-consistency](./endpoints/POST__api_v1_database_validate-consistency.md)
- [GET /api/v1/database/backup-list](./endpoints/GET__api_v1_database_backup-list.md)
- [POST /api/v1/database/schedule-backups](./endpoints/POST__api_v1_database_schedule-backups.md)
- [POST /api/v1/database/consistency-check](./endpoints/POST__api_v1_database_consistency-check.md)
- [POST /api/database/backup](./endpoints/POST__api_database_backup.md)
- [POST /api/database/migrate](./endpoints/POST__api_database_migrate.md)
- [GET /api/database/migration-info](./endpoints/GET__api_database_migration-info.md)
- [POST /api/database/validate-consistency](./endpoints/POST__api_database_validate-consistency.md)
- [GET /api/database/backup-list](./endpoints/GET__api_database_backup-list.md)
- [POST /api/database/schedule-backups](./endpoints/POST__api_database_schedule-backups.md)
- [POST /api/database/consistency-check](./endpoints/POST__api_database_consistency-check.md)

### Security
- [GET /api/v1/security/status](./endpoints/GET__api_v1_security_status.md)
- [POST /api/v1/security/validate/iam-role](./endpoints/POST__api_v1_security_validate_iam-role.md)
- [POST /api/v1/security/validate/s3-bucket](./endpoints/POST__api_v1_security_validate_s3-bucket.md)
- [POST /api/v1/security/validate/security-groups](./endpoints/POST__api_v1_security_validate_security-groups.md)
- [POST /api/v1/security/validate/vpc](./endpoints/POST__api_v1_security_validate_vpc.md)
- [GET /api/v1/security/validate/encryption-in-transit](./endpoints/GET__api_v1_security_validate_encryption-in-transit.md)
- [POST /api/v1/security/audit/comprehensive](./endpoints/POST__api_v1_security_audit_comprehensive.md)
- [GET /api/v1/security/compliance/report](./endpoints/GET__api_v1_security_compliance_report.md)
- [GET /api/security/status](./endpoints/GET__api_security_status.md)
- [POST /api/security/validate/iam-role](./endpoints/POST__api_security_validate_iam-role.md)
- [POST /api/security/validate/s3-bucket](./endpoints/POST__api_security_validate_s3-bucket.md)
- [POST /api/security/validate/security-groups](./endpoints/POST__api_security_validate_security-groups.md)
- [POST /api/security/validate/vpc](./endpoints/POST__api_security_validate_vpc.md)
- [GET /api/security/validate/encryption-in-transit](./endpoints/GET__api_security_validate_encryption-in-transit.md)
- [POST /api/security/audit/comprehensive](./endpoints/POST__api_security_audit_comprehensive.md)
- [GET /api/security/compliance/report](./endpoints/GET__api_security_compliance_report.md)

### Cost-Optimization
- [GET /api/v1/cost-optimization/estimate](./endpoints/GET__api_v1_cost-optimization_estimate.md)
- [GET /api/v1/cost-optimization/optimization](./endpoints/GET__api_v1_cost-optimization_optimization.md)
- [GET /api/v1/cost-optimization/compliance](./endpoints/GET__api_v1_cost-optimization_compliance.md)
- [POST /api/v1/cost-optimization/apply-s3-lifecycle/{bucket_name}](./endpoints/POST__api_v1_cost-optimization_apply-s3-lifecycle_bucket_name.md)
- [POST /api/v1/cost-optimization/setup-monitoring](./endpoints/POST__api_v1_cost-optimization_setup-monitoring.md)
- [GET /api/v1/cost-optimization/config](./endpoints/GET__api_v1_cost-optimization_config.md)
- [GET /api/cost-optimization/estimate](./endpoints/GET__api_cost-optimization_estimate.md)
- [GET /api/cost-optimization/optimization](./endpoints/GET__api_cost-optimization_optimization.md)
- [GET /api/cost-optimization/compliance](./endpoints/GET__api_cost-optimization_compliance.md)
- [POST /api/cost-optimization/apply-s3-lifecycle/{bucket_name}](./endpoints/POST__api_cost-optimization_apply-s3-lifecycle_bucket_name.md)
- [POST /api/cost-optimization/setup-monitoring](./endpoints/POST__api_cost-optimization_setup-monitoring.md)
- [GET /api/cost-optimization/config](./endpoints/GET__api_cost-optimization_config.md)

### Health
- [GET /api/v1/health](./endpoints/GET__api_v1_health.md)
- [GET /api/v1/health/detailed](./endpoints/GET__api_v1_health_detailed.md)
- [GET /api/v1/health/component/{component_name}](./endpoints/GET__api_v1_health_component_component_name.md)
- [GET /api/v1/metrics](./endpoints/GET__api_v1_metrics.md)
- [GET /api/v1/metrics/history](./endpoints/GET__api_v1_metrics_history.md)
- [GET /api/v1/alerts](./endpoints/GET__api_v1_alerts.md)
- [POST /api/v1/alerts/thresholds](./endpoints/POST__api_v1_alerts_thresholds.md)
- [GET /api/v1/status](./endpoints/GET__api_v1_status.md)
- [GET /api/health](./endpoints/GET__api_health.md)
- [GET /api/health/detailed](./endpoints/GET__api_health_detailed.md)
- [GET /api/health/component/{component_name}](./endpoints/GET__api_health_component_component_name.md)
- [GET /api/metrics](./endpoints/GET__api_metrics.md)
- [GET /api/metrics/history](./endpoints/GET__api_metrics_history.md)
- [GET /api/alerts](./endpoints/GET__api_alerts.md)
- [POST /api/alerts/thresholds](./endpoints/POST__api_alerts_thresholds.md)
- [GET /api/status](./endpoints/GET__api_status.md)

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
- [POST /api/backup/full](./endpoints/POST__api_backup_full.md)
- [POST /api/backup/database](./endpoints/POST__api_backup_database.md)
- [GET /api/backup/list](./endpoints/GET__api_backup_list.md)
- [GET /api/backup/download/{backup_name}](./endpoints/GET__api_backup_download_backup_name.md)
- [POST /api/backup/restore](./endpoints/POST__api_backup_restore.md)
- [POST /api/backup/upload](./endpoints/POST__api_backup_upload.md)
- [POST /api/export](./endpoints/POST__api_export.md)
- [GET /api/export/download/{export_name}](./endpoints/GET__api_export_download_export_name.md)
- [DELETE /api/backup/{backup_name}](./endpoints/DELETE__api_backup_backup_name.md)
- [POST /api/cleanup](./endpoints/POST__api_cleanup.md)
- [GET /api/storage/info](./endpoints/GET__api_storage_info.md)
- [POST /api/storage/cleanup](./endpoints/POST__api_storage_cleanup.md)

### Authentication
- [GET /auth/login](./endpoints/GET__auth_login.md)
- [POST /auth/login](./endpoints/POST__auth_login.md)
- [POST /auth/api/login](./endpoints/POST__auth_api_login.md)
- [POST /auth/logout](./endpoints/POST__auth_logout.md)
- [GET /auth/status](./endpoints/GET__auth_status.md)
- [GET /auth/stats](./endpoints/GET__auth_stats.md)


## Interactive Documentation
- **Swagger UI**: Available at `/docs` when running the API server
- **ReDoc**: Available at `/redoc` when running the API server

## Development
This documentation is automatically generated from the OpenAPI specification.
To regenerate, run:
```bash
python scripts/generate_docs.py --category api
```
