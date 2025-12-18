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
from typing import List, Dict, Tuple
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

def check_embedding_availability(db):
    """Check total embeddings available in database."""
    total_drawings = db.query(Drawing).count()
    total_embeddings = db.query(DrawingEmbedding).filter(
        DrawingEmbedding.embedding_vector.isnot(None)
    ).count()
    
    print(f"📊 Database embedding status:")
    print(f"  Total drawings: {total_drawings}")
    print(f"  Total embeddings: {total_embeddings}")
    print(f"  Coverage: {total_embeddings/total_drawings*100:.1f}%" if total_drawings > 0 else "  Coverage: 0%")
    
    # Check dimension distribution
    embeddings_sample = db.query(DrawingEmbedding).filter(
        DrawingEmbedding.embedding_vector.isnot(None)
    ).limit(100).all()
    
    if embeddings_sample:
        dimension_check = {}
        for emb in embeddings_sample:
            vector = pickle.loads(emb.embedding_vector)
            dim = len(np.array(vector))
            dimension_check[dim] = dimension_check.get(dim, 0) + 1
        
        print(f"  Dimension distribution (sample of {len(embeddings_sample)}):")
        for dim, count in sorted(dimension_check.items()):
            percentage = (count / len(embeddings_sample)) * 100
            status = "✓" if dim == 832 else "⚠"
            print(f"    {status} {dim}-dimensional: {count} embeddings ({percentage:.1f}%)")
    
    return total_embeddings

def get_embeddings_for_age_group(age_min: float, age_max: float, db) -> tuple:
    """Get hybrid embeddings for a specific age group from database (reusing existing embeddings)."""
    print(f"  🔄 Reloading hybrid embeddings from database for age group {age_min}-{age_max}...")
    
    # Use JOIN to get drawings with embeddings in one query (more efficient)
    query_results = db.query(Drawing, DrawingEmbedding).join(
        DrawingEmbedding, Drawing.id == DrawingEmbedding.drawing_id
    ).filter(
        Drawing.age_years >= age_min,
        Drawing.age_years < age_max,
        DrawingEmbedding.embedding_vector.isnot(None)
    ).all()
    
    print(f"  ✓ Found {len(query_results)} drawings with existing embeddings in age range")
    
    if len(query_results) == 0:
        raise ValueError(f"No existing embeddings found for age group {age_min}-{age_max}")
    
    # Process embeddings efficiently
    embeddings = []
    subject_counts = {}
    dimension_check = {}
    
    print(f"  ⏳ Loading {len(query_results)} pre-computed embeddings from database...")
    
    for i, (drawing, embedding_record) in enumerate(query_results, 1):
        # Show progress every 1000 embeddings or at the end
        if i % 1000 == 0 or i == len(query_results):
            print(f"    Progress: {i}/{len(query_results)} embeddings loaded ({i/len(query_results)*100:.1f}%)")
        
        # Deserialize the embedding vector (stored as binary)
        import pickle
        vector = pickle.loads(embedding_record.embedding_vector)
        vector_array = np.array(vector)
        
        # Track dimension for validation
        dim = len(vector_array)
        dimension_check[dim] = dimension_check.get(dim, 0) + 1
        
        # Track subject distribution
        subject = drawing.subject if drawing.subject else "unspecified"
        subject_counts[subject] = subject_counts.get(subject, 0) + 1
        
        embeddings.append(vector_array)
    
    print(f"  ✅ Successfully reloaded {len(embeddings)} embeddings from database")
    
    # Validate embedding dimensions
    print(f"  📊 Embedding dimension distribution:")
    for dim, count in sorted(dimension_check.items()):
        percentage = (count / len(embeddings)) * 100 if embeddings else 0
        status = "✓" if dim == 832 else "⚠"
        print(f"    {status} {dim}-dimensional: {count} embeddings ({percentage:.1f}%)")
    
    # Show subject stratification
    print(f"  📊 Subject category distribution:")
    for subject, count in sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        percentage = (count / len(embeddings)) * 100
        print(f"    {subject}: {count} drawings ({percentage:.1f}%)")
    
    if len(subject_counts) > 10:
        print(f"    ... and {len(subject_counts) - 10} more subject categories")
    
    # Warn if not all embeddings are hybrid format
    non_hybrid = sum(count for dim, count in dimension_check.items() if dim != 832)
    if non_hybrid > 0:
        print(f"  ⚠ WARNING: {non_hybrid} embeddings are not in hybrid format (832-dim)")
        print(f"    Run migration script: python scripts/migrate_to_subject_aware.py")
    
    return np.array(embeddings), subject_counts

def train_model_with_embeddings(embeddings: np.ndarray, age_min: float, age_max: float, 
                               config: TrainingConfig, subject_counts: Dict, db) -> Dict:
    """Train subject-aware autoencoder model directly with pre-loaded hybrid embeddings."""
    import json
    from pathlib import Path
    
    print(f"  🧠 Training subject-aware autoencoder on {len(embeddings)} hybrid embeddings...")
    print(f"  📐 Input dimension: {embeddings.shape[1]} (768 visual + 64 subject)")
    
    # Validate embedding dimensions
    if embeddings.shape[1] != 832:
        raise ValueError(f"Expected 832-dimensional hybrid embeddings, got {embeddings.shape[1]}")
    
    # Initialize trainer and train model (with verbose progress)
    trainer = AutoencoderTrainer(config, verbose=True)
    training_result = trainer.train(embeddings)
    
    # Calculate threshold (95th percentile of reconstruction errors)
    threshold = training_result["metrics"]["percentile_95"]
    
    # Prepare subject categories list
    supported_categories = list(subject_counts.keys())
    if "unspecified" not in supported_categories:
        supported_categories.append("unspecified")
    
    # Save model to database with subject-aware metadata
    model_params = {
        "training_config": config.__dict__,
        "architecture": training_result["model_architecture"],
        "training_metrics": training_result["metrics"],
        "training_history": training_result["training_history"],
        "subject_distribution": subject_counts,
        "embedding_type": "hybrid",
        "input_dimension": 832,
        "visual_dimension": 768,
        "subject_dimension": 64
    }
    
    age_group_model = AgeGroupModel(
        age_min=age_min,
        age_max=age_max,
        model_type="autoencoder",
        vision_model="vit",
        supports_subjects=True,
        subject_categories=json.dumps(supported_categories),
        embedding_type="hybrid",
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
        'training_config': config.__dict__,
        'subject_aware': True,
        'embedding_type': 'hybrid',
        'supported_subjects': supported_categories
    }, model_path)
    
    result = {
        "model_id": age_group_model.id,
        "age_range": (age_min, age_max),
        "sample_count": len(embeddings),
        "threshold": threshold,
        "training_result": training_result,
        "model_path": str(model_path),
        "subject_counts": subject_counts,
        "architecture_type": "subject_aware"
    }
    
    return result

def train_age_group_offline(age_min: float, age_max: float, min_samples: int, db) -> Dict:
    """Train a single subject-aware age group model offline using existing embeddings from database."""
    print(f"\n🔄 Training subject-aware model for age group {age_min}-{age_max} years (reusing existing embeddings)...")
    
    try:
        # Reload embeddings from database (no new embedding creation)
        embeddings, subject_counts = get_embeddings_for_age_group(age_min, age_max, db)
        
        if len(embeddings) < min_samples:
            print(f"  ✗ Insufficient data: {len(embeddings)} existing embeddings (need {min_samples})")
            return None
        
        print(f"  ✅ Reusing {len(embeddings)} pre-computed hybrid embeddings for subject-aware training")
        print(f"  📊 Subject diversity: {len(subject_counts)} different categories")
        
        # Check subject stratification balance
        if len(subject_counts) < 2:
            print(f"  ⚠ Warning: Only {len(subject_counts)} subject category found - limited subject diversity")
        
        # Check if model already exists
        existing_model = db.query(AgeGroupModel).filter(
            AgeGroupModel.age_min == age_min,
            AgeGroupModel.age_max == age_max,
            AgeGroupModel.is_active == True
        ).first()
        
        if existing_model:
            print(f"  ⚠ Model already exists (ID: {existing_model.id}). Deactivating...")
            existing_model.is_active = False
        
        # Create training configuration optimized for hybrid embeddings
        config = TrainingConfig(
            hidden_dims=[512, 256, 128, 64],  # Larger first layer for 832-dim input
            learning_rate=0.00001,  # Much lower learning rate for stability
            batch_size=32,   # Smaller batch size for more stable gradients
            epochs=100,      # More epochs to compensate for lower LR
            validation_split=0.2,
            early_stopping_patience=15  # More patience for slow convergence
        )
        
        print(f"  ⏳ Starting subject-aware training with {config.epochs} epochs...")
        print(f"  🏗️ Architecture: {config.hidden_dims} (optimized for 832-dim hybrid input)")
        start_time = time.time()
        
        # Train directly with our pre-loaded hybrid embeddings
        result = train_model_with_embeddings(
            embeddings=embeddings,
            age_min=age_min,
            age_max=age_max,
            config=config,
            subject_counts=subject_counts,
            db=db
        )
        
        training_time = time.time() - start_time
        print(f"  ✅ Subject-aware training completed in {training_time:.1f} seconds")
        print(f"  🆔 Model ID: {result['model_id']}")
        print(f"  🎯 Anomaly threshold: {result['threshold']:.4f}")
        print(f"  📉 Final validation loss: {result['training_result']['final_val_loss']:.6f}")
        print(f"  💾 Model saved to: {result['model_path']}")
        print(f"  📊 Database record: age_group_models table (ID: {result['model_id']})")
        print(f"  🎨 Subject categories: {len(result['subject_counts'])} supported")
        print(f"  🏗️ Architecture: Unified subject-aware autoencoder")
        
        return result
        
    except Exception as e:
        print(f"  ✗ Subject-aware training failed: {str(e)}")
        return None

def main():
    """Main subject-aware offline training workflow."""
    parser = argparse.ArgumentParser(description='Train subject-aware autoencoder models offline')
    parser.add_argument('--age-groups', type=str, 
                       help='Comma-separated age groups (e.g., "3-6,6-9,9-12")')
    parser.add_argument('--min-samples', type=int, default=10,
                       help='Minimum samples required per age group')
    parser.add_argument('--force', action='store_true',
                       help='Force retrain existing models')
    
    args = parser.parse_args()
    
    print("=== Subject-Aware Offline Model Training ===\n")
    print("🔄 Training unified subject-aware autoencoders (reusing existing embeddings)")
    print("📐 Input: 832-dimensional hybrid embeddings (768 visual + 64 subject)")
    print("🏗️ Architecture: Subject-aware autoencoder per age group")
    
    # Get database session
    try:
        db = get_database_session()
        print("✓ Database connection established")
    except Exception as e:
        print(f"✗ Failed to connect to database: {str(e)}")
        return
    
    # Check embedding availability
    total_embeddings = check_embedding_availability(db)
    if total_embeddings == 0:
        print("✗ No embeddings found in database. Run embedding generation first.")
        return
    
    print(f"✅ Found {total_embeddings} existing embeddings ready for training")
    
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
    
    print(f"\n📊 Training {len(age_groups)} subject-aware age groups: {age_groups}")
    
    # Train each age group
    successful_models = []
    failed_models = []
    total_subjects = set()
    
    total_start_time = time.time()
    
    for age_min, age_max in age_groups:
        result = train_age_group_offline(age_min, age_max, args.min_samples, db)
        
        if result:
            successful_models.append(result)
            # Collect all subject categories
            total_subjects.update(result['subject_counts'].keys())
        else:
            failed_models.append((age_min, age_max))
    
    total_time = time.time() - total_start_time
    
    # Summary
    print(f"\n=== Subject-Aware Training Summary ===")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Successful models: {len(successful_models)}")
    print(f"Failed models: {len(failed_models)}")
    print(f"Total subject categories: {len(total_subjects)}")
    
    if successful_models:
        print(f"\n✅ Successfully trained subject-aware models:")
        for result in successful_models:
            age_range = result['age_range']
            subject_count = len(result['subject_counts'])
            print(f"  - Age {age_range[0]}-{age_range[1]}: Model {result['model_id']} "
                  f"(threshold: {result['threshold']:.4f}, samples: {result['sample_count']}, subjects: {subject_count})")
    
    if failed_models:
        print(f"\n❌ Failed to train models:")
        for age_min, age_max in failed_models:
            print(f"  - Age {age_min}-{age_max}")
    
    # Show subject category overview
    if total_subjects:
        print(f"\n🎨 Subject Categories Across All Models:")
        sorted_subjects = sorted(total_subjects)
        for i, subject in enumerate(sorted_subjects[:10]):  # Show first 10
            print(f"  {i+1:2d}. {subject}")
        if len(sorted_subjects) > 10:
            print(f"  ... and {len(sorted_subjects) - 10} more categories")
    
    # Show storage locations
    if successful_models:
        print(f"\n💾 Model Storage Locations:")
        print(f"  📁 Filesystem: static/models/autoencoder_{{model_id}}.pkl")
        print(f"  🗄️  Database: age_group_models table (with subject_aware metadata)")
        print(f"  📊 Embeddings: drawing_embeddings table (832-dim hybrid format)")
        print(f"  🎯 Architecture: Unified subject-aware autoencoders")
    
    # Close database connection
    db.close()
    
    print(f"\n🎉 === Subject-Aware Training Complete ===")
    if successful_models:
        print("🚀 You can now start the server and test the subject-aware models:")
        print("   source venv/bin/activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        print("\n📱 Frontend (if needed):")
        print("   cd frontend && npm run dev")
        print("\n🔍 Check subject-aware models via API:")
        print("   curl http://localhost:8000/api/v1/models/")
        print("\n🎨 Test subject-aware analysis:")
        print("   Upload a drawing with subject category and see attribution results!")

if __name__ == "__main__":
    main()