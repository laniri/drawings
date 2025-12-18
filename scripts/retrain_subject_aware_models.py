#!/usr/bin/env python3
"""
Retrain all age-group models with subject-aware architecture.

This script directly retrains autoencoder models using the hybrid embeddings
that were created during the migration process.

Requirements: 12.2, 12.3
"""

import os
import sys
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add the app directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.database import Drawing, DrawingEmbedding, AgeGroupModel
from app.services.model_manager import ModelManager
from app.services.age_group_manager import AgeGroupManager
from app.utils.embedding_serialization import deserialize_embedding_from_db

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_hybrid_embeddings_for_age_group(age_min: float, age_max: float):
    """Get all hybrid embeddings for a specific age group."""
    DATABASE_URL = 'sqlite:///./drawings.db'
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    with SessionLocal() as db:
        # Get drawings in age range
        drawings = db.query(Drawing).filter(
            Drawing.age_years >= age_min,
            Drawing.age_years < age_max
        ).all()
        
        embeddings_data = []
        
        for drawing in drawings:
            # Get the hybrid embedding for this drawing
            embedding_record = db.query(DrawingEmbedding).filter(
                DrawingEmbedding.drawing_id == drawing.id,
                DrawingEmbedding.embedding_type == "hybrid"
            ).first()
            
            if embedding_record:
                try:
                    # Deserialize the hybrid embedding
                    hybrid_vector = deserialize_embedding_from_db(embedding_record.embedding_vector)
                    
                    if hybrid_vector is not None and len(hybrid_vector) == 832:
                        embeddings_data.append({
                            'drawing_id': drawing.id,
                            'age': drawing.age_years,
                            'subject': drawing.subject,
                            'embedding': hybrid_vector
                        })
                except Exception as e:
                    logger.warning(f"Failed to deserialize embedding for drawing {drawing.id}: {e}")
        
        logger.info(f"Found {len(embeddings_data)} valid hybrid embeddings for age group {age_min}-{age_max}")
        return embeddings_data


def retrain_age_group_model(age_min: float, age_max: float, min_samples: int = 50):
    """Retrain a single age group model with hybrid embeddings."""
    logger.info(f"Retraining model for age group {age_min}-{age_max} years")
    
    # Get hybrid embeddings for this age group
    embeddings_data = get_hybrid_embeddings_for_age_group(age_min, age_max)
    
    if len(embeddings_data) < min_samples:
        logger.warning(f"Insufficient data for age group {age_min}-{age_max}: {len(embeddings_data)} samples (minimum: {min_samples})")
        return None
    
    try:
        # Initialize model manager
        model_manager = ModelManager()
        
        # Extract embeddings and metadata
        embeddings = [data['embedding'] for data in embeddings_data]
        ages = [data['age'] for data in embeddings_data]
        subjects = [data['subject'] for data in embeddings_data]
        
        logger.info(f"Training with {len(embeddings)} hybrid embeddings (832-dim)")
        
        # Train the subject-aware model
        model_result = model_manager.train_age_group_model(
            embeddings=embeddings,
            ages=ages,
            subjects=subjects,
            age_min=age_min,
            age_max=age_max,
            model_type="autoencoder",
            vision_model="vit"
        )
        
        if model_result:
            logger.info(f"✓ Successfully trained model for age group {age_min}-{age_max}")
            logger.info(f"  Model ID: {model_result.get('model_id', 'N/A')}")
            logger.info(f"  Sample count: {model_result.get('sample_count', 'N/A')}")
            logger.info(f"  Threshold: {model_result.get('threshold', 'N/A')}")
            return model_result
        else:
            logger.error(f"✗ Failed to train model for age group {age_min}-{age_max}")
            return None
            
    except Exception as e:
        logger.error(f"✗ Exception during training for age group {age_min}-{age_max}: {e}")
        return None


def retrain_all_models():
    """Retrain all age group models with subject-aware architecture."""
    logger.info("Starting complete model retraining with subject-aware architecture")
    
    # Define age groups (same as in the training script)
    age_groups = [
        (2.0, 3.0, "Toddler"),
        (3.0, 4.0, "Early preschool"),
        (4.0, 5.0, "Late preschool"),
        (5.0, 6.0, "Kindergarten"),
        (6.0, 7.0, "Early elementary"),
        (7.0, 8.0, "Elementary"),
        (8.0, 9.0, "Late elementary"),
        (9.0, 12.0, "Middle childhood")
    ]
    
    successful_models = []
    failed_models = []
    
    for age_min, age_max, description in age_groups:
        logger.info(f"\n--- Training {description} model (ages {age_min}-{age_max}) ---")
        
        result = retrain_age_group_model(age_min, age_max, min_samples=30)
        
        if result:
            successful_models.append((age_min, age_max, description, result))
        else:
            failed_models.append((age_min, age_max, description))
    
    # Summary
    logger.info(f"\n=== Retraining Summary ===")
    logger.info(f"✓ Successful models: {len(successful_models)}")
    logger.info(f"✗ Failed models: {len(failed_models)}")
    
    if successful_models:
        logger.info("\nSuccessful models:")
        for age_min, age_max, description, result in successful_models:
            logger.info(f"  {description} ({age_min}-{age_max}): Model ID {result.get('model_id', 'N/A')}")
    
    if failed_models:
        logger.info("\nFailed models:")
        for age_min, age_max, description in failed_models:
            logger.info(f"  {description} ({age_min}-{age_max}): Training failed")
    
    return len(successful_models), len(failed_models)


def validate_retrained_models():
    """Validate that all retrained models are working correctly."""
    logger.info("Validating retrained models...")
    
    DATABASE_URL = 'sqlite:///./drawings.db'
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    with SessionLocal() as db:
        # Get all active models
        models = db.query(AgeGroupModel).filter(AgeGroupModel.is_active == True).all()
        
        logger.info(f"Found {len(models)} active models")
        
        valid_models = 0
        
        for model in models:
            # Check that model supports subjects and uses hybrid embeddings
            if (model.supports_subjects and 
                model.embedding_type == "hybrid" and 
                model.subject_categories is not None):
                valid_models += 1
                logger.info(f"  ✓ Model {model.id} (ages {model.age_min}-{model.age_max}): Valid subject-aware model")
            else:
                logger.warning(f"  ✗ Model {model.id} (ages {model.age_min}-{model.age_max}): Not properly configured for subject-aware architecture")
        
        logger.info(f"Validation complete: {valid_models}/{len(models)} models are properly configured")
        return valid_models == len(models)


def main():
    """Main retraining workflow."""
    logger.info("Starting subject-aware model retraining process")
    
    try:
        # Step 1: Retrain all models
        successful, failed = retrain_all_models()
        
        if successful == 0:
            logger.error("❌ No models were successfully retrained")
            sys.exit(1)
        
        # Step 2: Validate retrained models
        if validate_retrained_models():
            logger.info("✅ All retrained models are properly configured")
        else:
            logger.warning("⚠ Some models may not be properly configured")
        
        logger.info(f"\n🎉 Model retraining completed!")
        logger.info(f"   Successfully retrained: {successful} models")
        logger.info(f"   Failed: {failed} models")
        
        if failed > 0:
            logger.info("   Note: Failed models may be due to insufficient data for specific age groups")
        
    except Exception as e:
        logger.error(f"Retraining failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()