#!/usr/bin/env python3
"""
Database migration script for transitioning to subject-aware architecture.

This script:
1. Sets missing subject categories to "unspecified"
2. Converts existing embeddings to hybrid format (832-dim)
3. Updates analyses to include component-specific scores
4. Validates data integrity after migration

Requirements: 12.1
"""

import os
import sys
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json
import logging

# Add the app directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import get_db
from app.models.database import Drawing, DrawingEmbedding, AnomalyAnalysis, AgeGroupModel
from app.services.embedding_service import EmbeddingService
from app.schemas.drawings import SubjectCategory
from app.utils.embedding_serialization import serialize_embedding_for_db, deserialize_embedding_from_db

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def encode_subject_category_manual(subject: str) -> np.ndarray:
    """Manual implementation of subject category encoding."""
    # List of all supported subject categories
    categories = [
        "TV", "airplane", "apple", "bear", "bed", "bee", "bike", "bird", "boat", "book",
        "bottle", "bowl", "cactus", "camel", "car", "cat", "chair", "clock", "couch", "cow",
        "cup", "dog", "elephant", "face", "fish", "frog", "hand", "hat", "horse", "house",
        "ice cream", "key", "lamp", "mushroom", "octopus", "person", "phone", "piano",
        "rabbit", "scissors", "sheep", "snail", "spider", "tiger", "train", "tree",
        "watch", "whale", "unspecified"
    ]
    
    # Pad to 64 dimensions
    while len(categories) < 64:
        categories.append(f"reserved_{len(categories)}")
    
    # Create one-hot encoding
    encoding = np.zeros(64, dtype=np.float32)
    
    if subject in categories:
        index = categories.index(subject)
        encoding[index] = 1.0
    else:
        # Default to "unspecified"
        index = categories.index("unspecified")
        encoding[index] = 1.0
    
    return encoding


def check_embedding_dimensions():
    """Check the actual dimensions of existing embeddings."""
    DATABASE_URL = 'sqlite:///./drawings.db'
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    with SessionLocal() as db:
        # Sample a few embeddings to check their dimensions
        sample_embeddings = db.query(DrawingEmbedding).limit(10).all()
        
        dimensions = []
        for embedding in sample_embeddings:
            try:
                vector = deserialize_embedding_from_db(embedding.embedding_vector)
                dimensions.append(len(vector))
            except Exception as e:
                logger.error(f"Error deserializing embedding {embedding.id}: {e}")
                dimensions.append(0)
        
        logger.info(f"Sample embedding dimensions: {dimensions}")
        return dimensions


def migrate_drawings_subjects():
    """Set missing subject categories to 'unspecified'."""
    DATABASE_URL = 'sqlite:///./drawings.db'
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    with SessionLocal() as db:
        # Find drawings without subject
        drawings_without_subject = db.query(Drawing).filter(Drawing.subject.is_(None)).all()
        
        logger.info(f"Found {len(drawings_without_subject)} drawings without subject")
        
        # Update them to use "unspecified"
        for drawing in drawings_without_subject:
            drawing.subject = "unspecified"
            logger.info(f"Set drawing {drawing.id} subject to 'unspecified'")
        
        db.commit()
        logger.info("Successfully updated drawings with missing subjects")


def migrate_embeddings_to_hybrid():
    """Convert existing embeddings to hybrid format if needed."""
    DATABASE_URL = 'sqlite:///./drawings.db'
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    with SessionLocal() as db:
        # Get all embeddings
        embeddings = db.query(DrawingEmbedding).all()
        logger.info(f"Processing {len(embeddings)} embeddings")
        
        converted_count = 0
        error_count = 0
        
        for embedding in embeddings:
            try:
                # Deserialize the existing embedding
                vector = deserialize_embedding_from_db(embedding.embedding_vector)
                
                # Check if it's already hybrid format (832 dimensions)
                if len(vector) == 832:
                    # Already hybrid, just separate components if not already done
                    if embedding.visual_component is None or embedding.subject_component is None:
                        visual_component = vector[:768]
                        subject_component = vector[768:832]
                        
                        embedding.visual_component = serialize_embedding_for_db(visual_component)
                        embedding.subject_component = serialize_embedding_for_db(subject_component)
                        embedding.vector_dimension = 832
                        converted_count += 1
                        
                        if converted_count % 1000 == 0:
                            logger.info(f"Separated components for {converted_count} embeddings")
                    continue
                
                # If it's 768 dimensions (old ViT format), convert to hybrid
                if len(vector) == 768:
                    # Get the drawing to find its subject
                    drawing = db.query(Drawing).filter(Drawing.id == embedding.drawing_id).first()
                    if not drawing:
                        logger.error(f"Drawing not found for embedding {embedding.id}")
                        error_count += 1
                        continue
                    
                    # Ensure drawing has a subject
                    subject = drawing.subject if drawing.subject else "unspecified"
                    
                    # Create subject encoding
                    subject_encoding = encode_subject_category_manual(subject)
                    
                    # Create hybrid embedding
                    hybrid_vector = np.concatenate([vector, subject_encoding])
                    
                    # Update the embedding
                    embedding.embedding_vector = serialize_embedding_for_db(hybrid_vector)
                    embedding.visual_component = serialize_embedding_for_db(vector)
                    embedding.subject_component = serialize_embedding_for_db(subject_encoding)
                    embedding.vector_dimension = 832
                    embedding.embedding_type = "hybrid"
                    
                    converted_count += 1
                    
                    if converted_count % 1000 == 0:
                        logger.info(f"Converted {converted_count} embeddings to hybrid format")
                        db.commit()  # Commit periodically
                
                # If it's 769 dimensions (ViT + age), convert to hybrid by replacing age with subject
                elif len(vector) == 769:
                    # Get the drawing to find its subject
                    drawing = db.query(Drawing).filter(Drawing.id == embedding.drawing_id).first()
                    if not drawing:
                        logger.error(f"Drawing not found for embedding {embedding.id}")
                        error_count += 1
                        continue
                    
                    # Ensure drawing has a subject
                    subject = drawing.subject if drawing.subject else "unspecified"
                    
                    # Extract visual component (first 768 dimensions, removing age)
                    visual_component = vector[:768]
                    
                    # Create subject encoding
                    subject_encoding = encode_subject_category_manual(subject)
                    
                    # Create hybrid embedding
                    hybrid_vector = np.concatenate([visual_component, subject_encoding])
                    
                    # Update the embedding
                    embedding.embedding_vector = serialize_embedding_for_db(hybrid_vector)
                    embedding.visual_component = serialize_embedding_for_db(visual_component)
                    embedding.subject_component = serialize_embedding_for_db(subject_encoding)
                    embedding.vector_dimension = 832
                    embedding.embedding_type = "hybrid"
                    
                    converted_count += 1
                    
                    if converted_count % 1000 == 0:
                        logger.info(f"Converted {converted_count} embeddings to hybrid format")
                        db.commit()  # Commit periodically
                
                else:
                    logger.warning(f"Unexpected embedding dimension {len(vector)} for embedding {embedding.id}")
                    error_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing embedding {embedding.id}: {e}")
                error_count += 1
        
        db.commit()
        logger.info(f"Migration complete: {converted_count} embeddings converted, {error_count} errors")


def migrate_analyses_to_subject_aware():
    """Update existing analyses to include component-specific scores."""
    DATABASE_URL = 'sqlite:///./drawings.db'
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    with SessionLocal() as db:
        # Find analyses that don't have component scores
        analyses_to_update = db.query(AnomalyAnalysis).filter(
            AnomalyAnalysis.visual_anomaly_score.is_(None)
        ).all()
        
        logger.info(f"Found {len(analyses_to_update)} analyses to update with component scores")
        
        updated_count = 0
        
        for analysis in analyses_to_update:
            try:
                # For existing analyses, we can't recalculate the component scores
                # without re-running the analysis, so we'll set them to the overall score
                # and mark attribution as "unknown" until re-analysis
                analysis.visual_anomaly_score = analysis.anomaly_score
                analysis.subject_anomaly_score = 0.0  # Default to no subject anomaly
                analysis.anomaly_attribution = "visual"  # Default attribution
                analysis.analysis_type = "subject_aware"
                
                updated_count += 1
                
                if updated_count % 1000 == 0:
                    logger.info(f"Updated {updated_count} analyses")
                    db.commit()
                    
            except Exception as e:
                logger.error(f"Error updating analysis {analysis.id}: {e}")
        
        db.commit()
        logger.info(f"Updated {updated_count} analyses with component scores")


def update_age_group_models():
    """Update age group models to support subject-aware architecture."""
    DATABASE_URL = 'sqlite:///./drawings.db'
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    with SessionLocal() as db:
        models = db.query(AgeGroupModel).all()
        
        logger.info(f"Updating {len(models)} age group models")
        
        # List of supported subject categories
        supported_categories = [
            "TV", "airplane", "apple", "bear", "bed", "bee", "bike", "bird", "boat", "book",
            "bottle", "bowl", "cactus", "camel", "car", "cat", "chair", "clock", "couch", "cow",
            "cup", "dog", "elephant", "face", "fish", "frog", "hand", "hat", "horse", "house",
            "ice cream", "key", "lamp", "mushroom", "octopus", "person", "phone", "piano",
            "rabbit", "scissors", "sheep", "snail", "spider", "tiger", "train", "tree",
            "watch", "whale", "unspecified"
        ]
        
        for model in models:
            model.supports_subjects = True
            model.embedding_type = "hybrid"
            model.subject_categories = json.dumps(supported_categories)
        
        db.commit()
        logger.info("Updated all age group models for subject-aware architecture")


def validate_migration():
    """Validate the migration was successful."""
    DATABASE_URL = 'sqlite:///./drawings.db'
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    with SessionLocal() as db:
        # Check drawings
        drawings_without_subject = db.query(Drawing).filter(Drawing.subject.is_(None)).count()
        total_drawings = db.query(Drawing).count()
        
        # Check embeddings
        non_hybrid_embeddings = db.query(DrawingEmbedding).filter(
            DrawingEmbedding.embedding_type != 'hybrid'
        ).count()
        embeddings_without_components = db.query(DrawingEmbedding).filter(
            (DrawingEmbedding.visual_component.is_(None)) | 
            (DrawingEmbedding.subject_component.is_(None))
        ).count()
        total_embeddings = db.query(DrawingEmbedding).count()
        
        # Check analyses
        non_subject_aware_analyses = db.query(AnomalyAnalysis).filter(
            AnomalyAnalysis.analysis_type != 'subject_aware'
        ).count()
        analyses_without_components = db.query(AnomalyAnalysis).filter(
            AnomalyAnalysis.visual_anomaly_score.is_(None)
        ).count()
        total_analyses = db.query(AnomalyAnalysis).count()
        
        # Check models
        non_subject_models = db.query(AgeGroupModel).filter(
            AgeGroupModel.supports_subjects != True
        ).count()
        total_models = db.query(AgeGroupModel).count()
        
        logger.info("Migration Validation Results:")
        logger.info(f"  Drawings: {total_drawings} total, {drawings_without_subject} without subject")
        logger.info(f"  Embeddings: {total_embeddings} total, {non_hybrid_embeddings} non-hybrid, {embeddings_without_components} without components")
        logger.info(f"  Analyses: {total_analyses} total, {non_subject_aware_analyses} non-subject-aware, {analyses_without_components} without component scores")
        logger.info(f"  Models: {total_models} total, {non_subject_models} non-subject-aware")
        
        # Validation checks
        success = True
        if drawings_without_subject > 0:
            logger.error("VALIDATION FAILED: Some drawings still missing subjects")
            success = False
        
        if non_hybrid_embeddings > 0:
            logger.error("VALIDATION FAILED: Some embeddings not marked as hybrid")
            success = False
            
        if embeddings_without_components > 0:
            logger.error("VALIDATION FAILED: Some embeddings missing component separation")
            success = False
        
        if non_subject_aware_analyses > 0:
            logger.error("VALIDATION FAILED: Some analyses not marked as subject-aware")
            success = False
        
        if non_subject_models > 0:
            logger.error("VALIDATION FAILED: Some models not marked as subject-aware")
            success = False
        
        if success:
            logger.info("✓ Migration validation PASSED - All data successfully migrated")
        else:
            logger.error("✗ Migration validation FAILED - Some issues need to be resolved")
        
        return success


def main():
    """Run the complete migration process."""
    logger.info("Starting database migration to subject-aware architecture")
    
    try:
        # Step 1: Check current embedding dimensions
        logger.info("Step 1: Checking existing embedding dimensions")
        dimensions = check_embedding_dimensions()
        
        # Step 2: Migrate drawings subjects
        logger.info("Step 2: Migrating drawing subjects")
        migrate_drawings_subjects()
        
        # Step 3: Migrate embeddings to hybrid format
        logger.info("Step 3: Migrating embeddings to hybrid format")
        migrate_embeddings_to_hybrid()
        
        # Step 4: Migrate analyses to subject-aware format
        logger.info("Step 4: Migrating analyses to subject-aware format")
        migrate_analyses_to_subject_aware()
        
        # Step 5: Update age group models
        logger.info("Step 5: Updating age group models")
        update_age_group_models()
        
        # Step 6: Validate migration
        logger.info("Step 6: Validating migration")
        success = validate_migration()
        
        if success:
            logger.info("🎉 Database migration completed successfully!")
        else:
            logger.error("❌ Database migration completed with errors")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Migration failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()