"""add_performance_indexes

Revision ID: 4c0233b32be8
Revises: 9eec035faf01
Create Date: 2025-12-21 22:49:01.304801

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '4c0233b32be8'
down_revision = '9eec035faf01'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add performance indexes for common query patterns."""
    
    # === DRAWINGS TABLE INDEXES ===
    
    # 1. Age-based filtering (very common in list_drawings API)
    op.create_index('idx_drawings_age_years', 'drawings', ['age_years'])
    
    # 2. Upload timestamp for ordering (used in list_drawings with ORDER BY)
    op.create_index('idx_drawings_upload_timestamp', 'drawings', ['upload_timestamp'])
    
    # 3. Subject filtering (used in list_drawings API)
    op.create_index('idx_drawings_subject', 'drawings', ['subject'])
    
    # 4. Expert label filtering (used in list_drawings API)
    op.create_index('idx_drawings_expert_label', 'drawings', ['expert_label'])
    
    # 5. Composite index for age range queries with timestamp ordering
    op.create_index('idx_drawings_age_timestamp', 'drawings', ['age_years', 'upload_timestamp'])
    
    # 6. Composite index for filtered queries with pagination
    op.create_index('idx_drawings_subject_age_timestamp', 'drawings', ['subject', 'age_years', 'upload_timestamp'])
    
    
    # === ANOMALY_ANALYSES TABLE INDEXES ===
    
    # 7. Drawing ID lookup (very frequent - used in analysis results)
    op.create_index('idx_anomaly_analyses_drawing_id', 'anomaly_analyses', ['drawing_id'])
    
    # 8. Age group model lookup (used for model-specific queries)
    op.create_index('idx_anomaly_analyses_age_group_model_id', 'anomaly_analyses', ['age_group_model_id'])
    
    # 9. Anomaly flag filtering (dashboard queries for anomaly counts)
    op.create_index('idx_anomaly_analyses_is_anomaly', 'anomaly_analyses', ['is_anomaly'])
    
    # 10. Analysis timestamp for recent analyses
    op.create_index('idx_anomaly_analyses_timestamp', 'anomaly_analyses', ['analysis_timestamp'])
    
    # 11. Composite index for dashboard statistics (anomaly + timestamp)
    op.create_index('idx_anomaly_analyses_anomaly_timestamp', 'anomaly_analyses', ['is_anomaly', 'analysis_timestamp'])
    
    # 12. Normalized score for threshold calculations
    op.create_index('idx_anomaly_analyses_normalized_score', 'anomaly_analyses', ['normalized_score'])
    
    # 13. Composite index for threshold recalculation (model + score)
    op.create_index('idx_anomaly_analyses_model_score', 'anomaly_analyses', ['age_group_model_id', 'normalized_score'])
    
    
    # === DRAWING_EMBEDDINGS TABLE INDEXES ===
    
    # 14. Drawing ID lookup (used when retrieving embeddings for analysis)
    op.create_index('idx_drawing_embeddings_drawing_id', 'drawing_embeddings', ['drawing_id'])
    
    # 15. Model type filtering (when looking for specific embedding types)
    op.create_index('idx_drawing_embeddings_model_type', 'drawing_embeddings', ['model_type'])
    
    # 16. Embedding type filtering (hybrid vs other types)
    op.create_index('idx_drawing_embeddings_embedding_type', 'drawing_embeddings', ['embedding_type'])
    
    # 17. Composite index for embedding lookup (drawing + model + type)
    op.create_index('idx_drawing_embeddings_lookup', 'drawing_embeddings', ['drawing_id', 'model_type', 'embedding_type'])
    
    
    # === AGE_GROUP_MODELS TABLE INDEXES ===
    
    # 18. Active models filtering (only active models are used)
    op.create_index('idx_age_group_models_is_active', 'age_group_models', ['is_active'])
    
    # 19. Age range lookup (finding appropriate model for age)
    op.create_index('idx_age_group_models_age_range', 'age_group_models', ['age_min', 'age_max'])
    
    # 20. Composite index for active model lookup
    op.create_index('idx_age_group_models_active_age', 'age_group_models', ['is_active', 'age_min', 'age_max'])
    
    
    # === INTERPRETABILITY_RESULTS TABLE INDEXES ===
    
    # 21. Analysis ID lookup (used when retrieving interpretability for analysis)
    op.create_index('idx_interpretability_results_analysis_id', 'interpretability_results', ['analysis_id'])
    
    # 22. Creation timestamp for recent interpretability results
    op.create_index('idx_interpretability_results_timestamp', 'interpretability_results', ['created_timestamp'])
    
    
    # === TRAINING_JOBS TABLE INDEXES ===
    
    # 23. Status filtering (finding jobs by status)
    op.create_index('idx_training_jobs_status', 'training_jobs', ['status'])
    
    # 24. Environment filtering (local vs sagemaker)
    op.create_index('idx_training_jobs_environment', 'training_jobs', ['environment'])
    
    # 25. Start timestamp for job history
    op.create_index('idx_training_jobs_start_timestamp', 'training_jobs', ['start_timestamp'])
    
    
    # === TRAINING_REPORTS TABLE INDEXES ===
    
    # 26. Training job lookup
    op.create_index('idx_training_reports_training_job_id', 'training_reports', ['training_job_id'])
    
    # 27. Creation timestamp for recent reports
    op.create_index('idx_training_reports_timestamp', 'training_reports', ['created_timestamp'])


def downgrade() -> None:
    """Remove performance indexes."""
    
    # Drop all indexes in reverse order
    op.drop_index('idx_training_reports_timestamp')
    op.drop_index('idx_training_reports_training_job_id')
    op.drop_index('idx_training_jobs_start_timestamp')
    op.drop_index('idx_training_jobs_environment')
    op.drop_index('idx_training_jobs_status')
    op.drop_index('idx_interpretability_results_timestamp')
    op.drop_index('idx_interpretability_results_analysis_id')
    op.drop_index('idx_age_group_models_active_age')
    op.drop_index('idx_age_group_models_age_range')
    op.drop_index('idx_age_group_models_is_active')
    op.drop_index('idx_drawing_embeddings_lookup')
    op.drop_index('idx_drawing_embeddings_embedding_type')
    op.drop_index('idx_drawing_embeddings_model_type')
    op.drop_index('idx_drawing_embeddings_drawing_id')
    op.drop_index('idx_anomaly_analyses_model_score')
    op.drop_index('idx_anomaly_analyses_normalized_score')
    op.drop_index('idx_anomaly_analyses_anomaly_timestamp')
    op.drop_index('idx_anomaly_analyses_timestamp')
    op.drop_index('idx_anomaly_analyses_is_anomaly')
    op.drop_index('idx_anomaly_analyses_age_group_model_id')
    op.drop_index('idx_anomaly_analyses_drawing_id')
    op.drop_index('idx_drawings_subject_age_timestamp')
    op.drop_index('idx_drawings_age_timestamp')
    op.drop_index('idx_drawings_expert_label')
    op.drop_index('idx_drawings_subject')
    op.drop_index('idx_drawings_upload_timestamp')
    op.drop_index('idx_drawings_age_years')