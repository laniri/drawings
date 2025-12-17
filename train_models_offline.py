#!/usr/bin/env python3
"""
Offline model training script that trains models without requiring the server to be running.
This script directly accesses the database and trains models locally.
"""

import sys
import os
sys.path.append('.')

import json
import time
import argparse
import pickle
from pathlib import Path
from typing import List, Dict
import torch
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import app modules
from app.core.config import get_settings
from app.models.database import Drawing, AgeGroupModel, DrawingEmbedding
from app.services.model_manager import get_model_manager, TrainingConfig, AutoencoderTrainer
from app.services.embedding_service import get_embedding_service

def get_database_session():
    """Get database session directly."""
    settings = get_settings()
    engine = create_engine(settings.DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()

def get_embeddings_for_age_group(age_min: float, age_max: float, db) -> List[np.ndarray]:
    """Get embeddings for a specific age group."""
    print(f"  Fetching embeddings for age group {age_min}-{age_max}...")
    
    # Get drawings in age range
    drawings = db.query(Drawing).filter(
        Drawing.age_years >= age_min,
        Drawing.age_years < age_max
    ).all()
    
    print(f"  Found {len(drawings)} drawings in age range")
    
    # Get embeddings for these drawings
    embeddings = []
    missing_embeddings = 0
    
    print(f"  ⏳ Retrieving embeddings for {len(drawings)} drawings...")
    
    for i, drawing in enumerate(drawings, 1):
        # Show progress every 1000 drawings or at the end
        if i % 1000 == 0 or i == len(drawings):
            print(f"    Progress: {i}/{len(drawings)} drawings processed ({i/len(drawings)*100:.1f}%)")
        
        embedding_record = db.query(DrawingEmbedding).filter(
            DrawingEmbedding.drawing_id == drawing.id
        ).first()
        
        if embedding_record and embedding_record.embedding_vector:
            # Deserialize the embedding vector (stored as binary)
            import pickle
            vector = pickle.loads(embedding_record.embedding_vector)
            embeddings.append(np.array(vector))
        else:
            missing_embeddings += 1
    
    print(f"  ✓ Retrieved {len(embeddings)} embeddings ({missing_embeddings} missing)")
    
    if missing_embeddings > 0:
        print(f"  ⚠ Warning: {missing_embeddings} drawings missing embeddings ({missing_embeddings/len(drawings)*100:.1f}%)")
    
    if len(embeddings) == 0:
        raise ValueError(f"No embeddings found for age group {age_min}-{age_max}")
    
    return np.array(embeddings)

def train_model_with_embeddings(embeddings: np.ndarray, age_min: float, age_max: float, 
                               config: TrainingConfig, db) -> Dict:
    """Train autoencoder model directly with pre-loaded embeddings."""
    import json
    from pathlib import Path
    
    # Initialize trainer and train model (with verbose progress)
    trainer = AutoencoderTrainer(config, verbose=True)
    training_result = trainer.train(embeddings)
    
    # Calculate threshold (95th percentile of reconstruction errors)
    threshold = training_result["metrics"]["percentile_95"]
    
    # Save model to database
    model_params = {
        "training_config": config.__dict__,
        "architecture": training_result["model_architecture"],
        "training_metrics": training_result["metrics"],
        "training_history": training_result["training_history"]
    }
    
    age_group_model = AgeGroupModel(
        age_min=age_min,
        age_max=age_max,
        model_type="autoencoder",
        vision_model="vit",
        parameters=json.dumps(model_params),
        sample_count=len(embeddings),
        threshold=threshold
    )
    
    db.add(age_group_model)
    db.commit()
    db.refresh(age_group_model)
    
    # Save model weights
    models_dir = Path("static/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"autoencoder_{age_group_model.id}.pkl"
    
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'model_architecture': trainer.model.get_architecture_info(),
        'training_config': config.__dict__
    }, model_path)
    
    result = {
        "model_id": age_group_model.id,
        "age_range": (age_min, age_max),
        "sample_count": len(embeddings),
        "threshold": threshold,
        "training_result": training_result,
        "model_path": str(model_path)
    }
    
    return result

def train_age_group_offline(age_min: float, age_max: float, min_samples: int, db) -> Dict:
    """Train a single age group model offline."""
    print(f"\nTraining model for age group {age_min}-{age_max} years...")
    
    try:
        # Get embeddings
        embeddings = get_embeddings_for_age_group(age_min, age_max, db)
        
        if len(embeddings) < min_samples:
            print(f"  ✗ Insufficient data: {len(embeddings)} samples (need {min_samples})")
            return None
        
        print(f"  ✓ Using {len(embeddings)} embeddings for training")
        
        # Check if model already exists
        existing_model = db.query(AgeGroupModel).filter(
            AgeGroupModel.age_min == age_min,
            AgeGroupModel.age_max == age_max,
            AgeGroupModel.is_active == True
        ).first()
        
        if existing_model:
            print(f"  ⚠ Model already exists (ID: {existing_model.id}). Deactivating...")
            existing_model.is_active = False
        
        # Create training configuration (very stable for offline training)
        config = TrainingConfig(
            hidden_dims=[256, 128, 64, 32],
            learning_rate=0.00001,  # Much lower learning rate for stability
            batch_size=32,   # Smaller batch size for more stable gradients
            epochs=100,      # More epochs to compensate for lower LR
            validation_split=0.2,
            early_stopping_patience=15  # More patience for slow convergence
        )
        
        print(f"  ⏳ Starting training with {config.epochs} epochs...")
        start_time = time.time()
        
        # Train directly with our pre-loaded embeddings (more efficient)
        result = train_model_with_embeddings(
            embeddings=embeddings,
            age_min=age_min,
            age_max=age_max,
            config=config,
            db=db
        )
        
        training_time = time.time() - start_time
        print(f"  ✅ Training completed in {training_time:.1f} seconds")
        print(f"  🆔 Model ID: {result['model_id']}")
        print(f"  🎯 Anomaly threshold: {result['threshold']:.4f}")
        print(f"  📉 Final validation loss: {result['training_result']['final_val_loss']:.6f}")
        print(f"  💾 Model saved to: {result['model_path']}")
        print(f"  📊 Database record: age_group_models table (ID: {result['model_id']})")
        
        return result
        
    except Exception as e:
        print(f"  ✗ Training failed: {str(e)}")
        return None

def main():
    """Main offline training workflow."""
    parser = argparse.ArgumentParser(description='Train autoencoder models offline')
    parser.add_argument('--age-groups', type=str, 
                       help='Comma-separated age groups (e.g., "3-6,6-9,9-12")')
    parser.add_argument('--min-samples', type=int, default=10,
                       help='Minimum samples required per age group')
    parser.add_argument('--force', action='store_true',
                       help='Force retrain existing models')
    
    args = parser.parse_args()
    
    print("=== Offline Model Training ===\n")
    
    # Get database session
    try:
        db = get_database_session()
        print("✓ Database connection established")
    except Exception as e:
        print(f"✗ Failed to connect to database: {str(e)}")
        return
    
    # Define age groups
    if args.age_groups:
        age_groups = []
        for group_str in args.age_groups.split(','):
            parts = group_str.strip().split('-')
            if len(parts) != 2:
                print(f"✗ Invalid age group format: {group_str}")
                return
            age_min, age_max = float(parts[0]), float(parts[1])
            age_groups.append((age_min, age_max))
    else:
        # Default age groups
        age_groups = [
            (2.0, 3.0),   # Early childhood
            (3.0, 4.0),   # Early childhood
            (4.0, 5.0),   # Early childhood
            (5.0, 6.0),   # Early childhood
            (6.0, 7.0),   # Early childhood
            (7.0, 8.0),   # Early childhood
            (8.0, 9.0),   # Middle childhood  
            (9.0, 12.0)   # Late childhood
        ]
    
    print(f"Training {len(age_groups)} age groups: {age_groups}")
    
    # Train each age group
    successful_models = []
    failed_models = []
    
    total_start_time = time.time()
    
    for age_min, age_max in age_groups:
        result = train_age_group_offline(age_min, age_max, args.min_samples, db)
        
        if result:
            successful_models.append(result)
        else:
            failed_models.append((age_min, age_max))
    
    total_time = time.time() - total_start_time
    
    # Summary
    print(f"\n=== Training Summary ===")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Successful models: {len(successful_models)}")
    print(f"Failed models: {len(failed_models)}")
    
    if successful_models:
        print(f"\n✅ Successfully trained models:")
        for result in successful_models:
            age_range = result['age_range']
            print(f"  - Age {age_range[0]}-{age_range[1]}: Model {result['model_id']} "
                  f"(threshold: {result['threshold']:.4f}, samples: {result['sample_count']})")
    
    if failed_models:
        print(f"\n❌ Failed to train models:")
        for age_min, age_max in failed_models:
            print(f"  - Age {age_min}-{age_max}")
    
    # Show storage locations
    if successful_models:
        print(f"\n💾 Model Storage Locations:")
        print(f"  📁 Filesystem: static/models/autoencoder_{{model_id}}.pkl")
        print(f"  🗄️  Database: age_group_models table")
        print(f"  📊 Embeddings: drawing_embeddings table")
    
    # Close database connection
    db.close()
    
    print(f"\n🎉 === Training Complete ===")
    if successful_models:
        print("🚀 You can now start the server and test the trained models:")
        print("   source venv/bin/activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        print("\n📱 Frontend (if needed):")
        print("   cd frontend && npm run dev")
        print("\n🔍 Check models via API:")
        print("   curl http://localhost:8000/api/v1/models/")

if __name__ == "__main__":
    main()