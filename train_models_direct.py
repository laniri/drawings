#!/usr/bin/env python3
"""
Direct local training script that bypasses the API for large datasets.
This script trains models directly using the local training environment.
"""

import sys
import os
sys.path.append('.')

import time
from pathlib import Path
from typing import List, Dict
import numpy as np
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.database import Drawing, DrawingEmbedding, AgeGroupModel
from app.services.local_training_environment import get_local_training_environment
from app.services.training_config import TrainingConfig, ModelConfig, OptimizerConfig, DataConfig
from app.services.model_manager import get_model_manager
from app.utils.embedding_serialization import deserialize_embedding_from_db

def get_embeddings_for_age_group(age_min: float, age_max: float, db: Session) -> np.ndarray:
    """Retrieve embeddings for a specific age group directly from database."""
    print(f"  Retrieving embeddings for age range {age_min}-{age_max}...")
    
    # Query drawings in age range
    drawings = db.query(Drawing).filter(
        Drawing.age_years >= age_min,
        Drawing.age_years <= age_max
    ).all()
    
    if not drawings:
        raise ValueError(f"No drawings found for age range {age_min}-{age_max}")
    
    print(f"  Found {len(drawings)} drawings in age range")
    
    # Get embeddings for these drawings
    embeddings = []
    missing_embeddings = 0
    
    for drawing in drawings:
        # Get the most recent embedding for this drawing
        embedding_record = db.query(DrawingEmbedding).filter(
            DrawingEmbedding.drawing_id == drawing.id
        ).order_by(DrawingEmbedding.created_timestamp.desc()).first()
        
        if embedding_record:
            # Deserialize embedding
            embedding_data = deserialize_embedding_from_db(embedding_record.embedding_vector)
            embeddings.append(embedding_data)
        else:
            missing_embeddings += 1
    
    if missing_embeddings > 0:
        print(f"  ⚠ Warning: {missing_embeddings} drawings missing embeddings")
    
    if not embeddings:
        raise ValueError(f"No embeddings found for age range {age_min}-{age_max}")
    
    print(f"  ✓ Retrieved {len(embeddings)} embeddings")
    return np.array(embeddings)

def train_age_group_direct(age_min: float, age_max: float, description: str, db: Session) -> Dict:
    """Train a single age group model directly."""
    print(f"\nTraining model for {description} (ages {age_min}-{age_max})...")
    
    try:
        # Get embeddings
        embeddings = get_embeddings_for_age_group(age_min, age_max, db)
        
        # Create training configuration
        config = TrainingConfig(
            job_name=f"direct_training_{description.lower().replace(' ', '_')}",
            epochs=100,
            model=ModelConfig(
                hidden_dims=[512, 256, 128, 64],
                latent_dim=32,
                dropout_rate=0.1
            ),
            optimizer=OptimizerConfig(
                learning_rate=0.001,
                weight_decay=1e-5
            ),
            data=DataConfig(
                batch_size=32,
                validation_split=0.2
            ),
            early_stopping_patience=15,
            save_plots=True
        )
        
        # Train using model manager
        model_manager = get_model_manager()
        result = model_manager.train_age_group_model(
            age_min=age_min,
            age_max=age_max,
            config=config,
            db=db
        )
        
        print(f"  ✓ Training completed successfully!")
        print(f"    Model ID: {result['model_id']}")
        print(f"    Sample count: {result['sample_count']}")
        print(f"    Threshold: {result['threshold']:.6f}")
        
        return result
        
    except Exception as e:
        print(f"  ✗ Training failed: {str(e)}")
        return {"error": str(e)}

def main():
    """Main direct training workflow."""
    print("=== Direct Local Training - Children's Drawing Anomaly Detection ===\n")
    
    # Get database session
    db = next(get_db())
    
    # Check total embeddings
    total_embeddings = db.query(DrawingEmbedding).count()
    total_drawings = db.query(Drawing).count()
    
    print(f"Database status:")
    print(f"  Total drawings: {total_drawings}")
    print(f"  Total embeddings: {total_embeddings}")
    
    if total_embeddings == 0:
        print("❌ No embeddings found. Please generate embeddings first:")
        print("  python train_models.py  # Run embedding generation part")
        return
    
    # Define age groups
    age_groups = [
        (3.0, 6.0, "Early childhood"),
        (6.0, 9.0, "Middle childhood"),
        (9.0, 12.0, "Late childhood")
    ]
    
    # Train each age group
    results = []
    start_time = time.time()
    
    for age_min, age_max, description in age_groups:
        result = train_age_group_direct(age_min, age_max, description, db)
        results.append(result)
    
    # Summary
    total_time = time.time() - start_time
    successful_models = [r for r in results if "model_id" in r]
    failed_models = [r for r in results if "error" in r]
    
    print(f"\n=== Training Summary ===")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Successful models: {len(successful_models)}")
    print(f"Failed models: {len(failed_models)}")
    
    if successful_models:
        print(f"\n✅ Successfully trained models:")
        for result in successful_models:
            print(f"  Model {result['model_id']}: ages {result['age_range'][0]}-{result['age_range'][1]}, "
                  f"samples: {result['sample_count']}, threshold: {result['threshold']:.6f}")
    
    if failed_models:
        print(f"\n❌ Failed models:")
        for result in failed_models:
            print(f"  Error: {result['error']}")
    
    # Test anomaly detection if models were created
    if successful_models:
        print(f"\n🧪 Testing anomaly detection...")
        try:
            import requests
            response = requests.post("http://localhost:8000/api/v1/analysis/analyze/1", timeout=30)
            if response.status_code == 200:
                result = response.json()
                print(f"  ✓ Test successful! Anomaly score: {result.get('anomaly_score', 'N/A')}")
            else:
                print(f"  ⚠ Test failed: {response.status_code}")
        except Exception as e:
            print(f"  ⚠ Test error: {str(e)}")
    
    print(f"\n=== Training Complete ===")
    print(f"Next steps:")
    print(f"1. Check models: curl http://localhost:8000/api/v1/models/age-groups")
    print(f"2. Open web interface: http://localhost:5173")
    print(f"3. Test analysis: curl -X POST http://localhost:8000/api/v1/analysis/analyze/1")

if __name__ == "__main__":
    main()