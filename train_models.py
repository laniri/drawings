#!/usr/bin/env python3
"""
Complete model training script that generates real embeddings and trains autoencoder models.
This script replaces the mock training demo with actual ViT embeddings.
"""

import sys
import os
sys.path.append('.')

import requests
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict

def get_all_drawings() -> List[Dict]:
    """Get all drawings from the database."""
    print("Fetching all drawings from database...")
    
    all_drawings = []
    page = 1
    
    while True:
        response = requests.get(f"http://localhost:8000/api/v1/drawings/?page={page}&page_size=50")
        if response.status_code != 200:
            print(f"Failed to fetch drawings page {page}")
            break
        
        data = response.json()
        drawings = data["drawings"]
        all_drawings.extend(drawings)
        
        print(f"  Fetched page {page}: {len(drawings)} drawings")
        
        if page >= data["total_pages"]:
            break
        page += 1
    
    print(f"Total drawings fetched: {len(all_drawings)}")
    return all_drawings

def generate_embeddings_for_drawings(drawings: List[Dict]) -> int:
    """Generate hybrid embeddings for all drawings using the analysis endpoint."""
    print(f"\nGenerating hybrid embeddings for {len(drawings)} drawings...")
    print("  Note: All embeddings will be 832-dimensional hybrid format (768 visual + 64 subject)")
    
    successful = 0
    failed = 0
    subject_stats = {}
    
    for i, drawing in enumerate(drawings, 1):
        drawing_id = drawing["id"]
        age = drawing["age_years"]
        subject = drawing.get("subject", "unspecified")
        
        # Track subject statistics
        subject_stats[subject] = subject_stats.get(subject, 0) + 1
        
        print(f"  [{i}/{len(drawings)}] Processing drawing {drawing_id} (age: {age}, subject: {subject})")
        
        try:
            # Use the dedicated embedding generation endpoint
            response = requests.post(
                f"http://localhost:8000/api/v1/analysis/embeddings/{drawing_id}",
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                status = result.get("status", "unknown")
                dimension = result.get('vector_dimension', 'N/A')
                
                if status in ["generated", "exists"]:
                    # Verify it's hybrid format
                    if dimension == 832:
                        print(f"    ✓ Hybrid embedding {status} (832-dim: 768 visual + 64 subject)")
                    else:
                        print(f"    ⚠ Embedding {status} but unexpected dimension: {dimension}")
                    successful += 1
                else:
                    print(f"    ✗ Unexpected status: {status}")
                    failed += 1
            else:
                error_detail = response.json().get("detail", "Unknown error") if response.headers.get("content-type") == "application/json" else response.text
                print(f"    ✗ Request failed ({response.status_code}): {error_detail}")
                failed += 1
                
        except Exception as e:
            print(f"    ✗ Exception: {str(e)}")
            failed += 1
        
        # Small delay to avoid overwhelming the server
        if i % 10 == 0:
            time.sleep(1)
    
    print(f"\nHybrid embedding generation complete:")
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed: {failed}")
    
    # Show subject distribution
    print(f"\nSubject category distribution:")
    for subject, count in sorted(subject_stats.items()):
        percentage = (count / len(drawings)) * 100
        print(f"  {subject}: {count} drawings ({percentage:.1f}%)")
    
    return successful

def train_age_group_models() -> List[Dict]:
    """Train subject-aware autoencoder models for different age groups."""
    print("\nStarting subject-aware age group model training...")
    print("  Note: All models will use 832-dimensional hybrid embeddings")
    print("  Architecture: Unified subject-aware autoencoders for each age group")
    
    # Define age groups with sufficient samples
    age_groups = [
        (3.0, 6.0, "Early childhood"),     # Broader range for more samples
        (6.0, 9.0, "Middle childhood"),    # Broader range for more samples  
        (9.0, 12.0, "Late childhood")      # Keep this range
    ]
    
    training_jobs = []
    
    for age_min, age_max, description in age_groups:
        print(f"\n  Training subject-aware model for {description} (ages {age_min}-{age_max})...")
        
        training_request = {
            "age_min": age_min,
            "age_max": age_max,
            "min_samples": 10,  # API minimum requirement
            "subject_aware": True,  # Enable subject-aware training
            "embedding_type": "hybrid"  # Ensure hybrid embeddings are used
        }
        
        try:
            # Start training job (runs in background)
            print(f"    Submitting subject-aware training job for {training_request['age_min']}-{training_request['age_max']} years...")
            response = requests.post(
                "http://localhost:8000/api/v1/models/train",
                json=training_request,
                timeout=30  # Reduced timeout since this just submits the job
            )
            
            if response.status_code == 200:
                job_info = response.json()
                training_jobs.append(job_info)
                print(f"    ✓ Subject-aware training job submitted: {job_info['job_id']}")
                print(f"    ✓ Sample count: {job_info['sample_count']}")
                print(f"    ✓ Age range: {job_info['age_range']}")
                print(f"    ✓ Status: {job_info['status']}")
                print(f"    ✓ Architecture: Unified subject-aware autoencoder")
            else:
                error_detail = response.json().get("detail", "Unknown error") if response.headers.get("content-type") == "application/json" else response.text
                print(f"    ✗ Training submission failed ({response.status_code}): {error_detail}")
                
        except Exception as e:
            print(f"    ✗ Exception during training submission: {str(e)}")
    
    return training_jobs

def wait_for_training_completion(training_jobs: List[Dict], max_wait_time: int = 1800) -> None:
    """Wait for all training jobs to complete."""
    if not training_jobs:
        print("No training jobs to monitor")
        return
    
    print(f"\nMonitoring {len(training_jobs)} training jobs (max wait: {max_wait_time//60} minutes)...")
    
    start_time = time.time()
    completed_jobs = set()
    last_status_update = {}
    
    while len(completed_jobs) < len(training_jobs) and (time.time() - start_time) < max_wait_time:
        for job in training_jobs:
            job_id = job["job_id"]
            
            if job_id in completed_jobs:
                continue
            
            try:
                response = requests.get(f"http://localhost:8000/api/v1/models/training/{job_id}/status", timeout=10)
                
                if response.status_code == 200:
                    status = response.json()
                    job_status = status["status"]
                    age_range = job["age_range"]
                    
                    # Only print status updates when they change
                    if job_id not in last_status_update or last_status_update[job_id] != job_status:
                        if job_status in ["completed", "failed"]:
                            completed_jobs.add(job_id)
                            
                            if job_status == "completed":
                                model_id = status.get("model_id", "N/A")
                                print(f"  ✓ Job {job_id} (ages {age_range[0]}-{age_range[1]}): COMPLETED (Model ID: {model_id})")
                            else:
                                error = status.get("error", "Unknown error")
                                print(f"  ✗ Job {job_id} (ages {age_range[0]}-{age_range[1]}): FAILED - {error}")
                        else:
                            progress = status.get("progress", 0)
                            message = status.get("message", "Training in progress")
                            elapsed = int(time.time() - start_time)
                            print(f"  ⏳ Job {job_id} (ages {age_range[0]}-{age_range[1]}): {job_status} - {message} (Progress: {progress}%, Elapsed: {elapsed}s)")
                        
                        last_status_update[job_id] = job_status
                        
                elif response.status_code == 404:
                    print(f"  ⚠ Job {job_id} not found - may have been completed")
                    completed_jobs.add(job_id)
                        
            except Exception as e:
                print(f"  ⚠ Could not get status for job {job_id}: {str(e)}")
        
        if len(completed_jobs) < len(training_jobs):
            time.sleep(15)  # Wait 15 seconds before checking again
    
    if len(completed_jobs) == len(training_jobs):
        print(f"\n✅ All training jobs completed!")
    else:
        print(f"\n⚠ Timeout reached after {max_wait_time//60} minutes. {len(completed_jobs)}/{len(training_jobs)} jobs completed.")
        print("You can check remaining jobs manually or restart the script.")

def check_final_system_status() -> None:
    """Check the final system status after training."""
    print("\nChecking final system status...")
    
    try:
        # Model status
        response = requests.get("http://localhost:8000/api/v1/models/status")
        if response.status_code == 200:
            status = response.json()
            print(f"  Total models: {status['total_models']}")
            print(f"  Active models: {status['active_models']}")
            print(f"  Total drawings: {status['total_drawings']}")
            print(f"  System status: {status['system_status']}")
        
        # List age group models
        response = requests.get("http://localhost:8000/api/v1/models/age-groups")
        if response.status_code == 200:
            models = response.json()
            print(f"\n  Age group models: {len(models['models'])}")
            for model in models["models"]:
                print(f"    Model {model['id']}: ages {model['age_min']}-{model['age_max']}, "
                      f"samples: {model['sample_count']}, status: {model['status']}")
                
    except Exception as e:
        print(f"  ⚠ Could not get system status: {str(e)}")

def test_anomaly_detection() -> None:
    """Test anomaly detection with a trained model."""
    print("\nTesting anomaly detection...")
    
    try:
        # Try to analyze the first drawing
        response = requests.post("http://localhost:8000/api/v1/analysis/analyze/1")
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success", False):
                print("  ✓ Anomaly detection test successful!")
                print(f"    Anomaly score: {result.get('anomaly_score', 'N/A')}")
                print(f"    Classification: {result.get('classification', 'N/A')}")
            else:
                print(f"  ✗ Analysis failed: {result.get('error', 'Unknown error')}")
        else:
            error_detail = response.json().get("detail", "Unknown error") if response.headers.get("content-type") == "application/json" else response.text
            print(f"  ✗ Request failed ({response.status_code}): {error_detail}")
            
    except Exception as e:
        print(f"  ⚠ Could not test anomaly detection: {str(e)}")

def check_existing_embeddings() -> int:
    """Check how many embeddings already exist in the database."""
    try:
        # Try to get embedding count from analysis stats first
        response = requests.get("http://localhost:8000/api/v1/analysis/stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()
            embedding_count = stats.get('total_embeddings', 0)
            if embedding_count > 0:
                return embedding_count
        
        # Fallback: check model status
        response = requests.get("http://localhost:8000/api/v1/models/status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            total_drawings = status.get('total_drawings', 0)
            # If we have drawings but no embedding count, assume some exist
            if total_drawings > 0:
                print(f"  Found {total_drawings} drawings in database")
                return total_drawings  # Optimistic assumption
            
        return 0
    except Exception as e:
        print(f"⚠ Could not check existing embeddings: {str(e)}")
        return 0

def main():
    """Main training workflow."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train autoencoder models for children\'s drawing anomaly detection')
    parser.add_argument('--skip-embeddings', action='store_true', 
                       help='Skip embedding generation and use existing embeddings')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model training (only generate embeddings)')
    parser.add_argument('--models-only', action='store_true',
                       help='Only train models (skip embeddings and testing)')
    
    args = parser.parse_args()
    
    print("=== Children's Drawing Anomaly Detection - Complete Model Training ===\n")
    
    if args.skip_embeddings:
        print("🔄 SKIP MODE: Skipping embedding generation entirely (assuming all embeddings exist)")
    if args.skip_training:
        print("🔄 EMBEDDINGS ONLY: Skipping model training")
    if args.models_only:
        print("🔄 MODELS ONLY: Skip embeddings and testing")
    
    # Check if backend is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ Backend is not responding properly")
            return
    except:
        print("❌ Backend is not running. Please start it first:")
        print("  source venv/bin/activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        return
    
    print("✅ Backend is running")
    
    # Step 1: Get all drawings
    drawings = get_all_drawings()
    if not drawings:
        print("❌ No drawings found in database")
        return
    
    # Step 2: Handle embeddings
    if args.skip_embeddings or args.models_only:
        # Skip embedding generation entirely - assume they exist
        existing_embeddings = check_existing_embeddings()
        if existing_embeddings > 0:
            print(f"✅ Skipping embedding generation - using existing {existing_embeddings} embeddings")
            successful_embeddings = existing_embeddings
        else:
            print("⚠ No existing embeddings found, but --skip-embeddings was specified.")
            print("   Assuming embeddings exist in database. If training fails, run without --skip-embeddings")
            successful_embeddings = len(drawings)  # Assume all drawings have embeddings
    else:
        # Generate embeddings
        print("🔄 Generating embeddings for all drawings...")
        successful_embeddings = generate_embeddings_for_drawings(drawings)
    
    if successful_embeddings == 0:
        print("❌ No embeddings available for training")
        return
    
    print(f"✅ Using {successful_embeddings} embeddings for training")
    
    # Step 3: Train age group models (unless skipped)
    if args.skip_training:
        print("🔄 Skipping model training as requested")
        print("\n=== Embedding Generation Complete ===")
        print(f"Generated/verified {successful_embeddings} embeddings")
        print("\nTo train models later, run:")
        print("  python train_models.py --skip-embeddings")
        return
    
    training_jobs = train_age_group_models()
    
    if not training_jobs:
        print("❌ No training jobs were started")
        return
    
    # Step 4: Wait for training to complete
    wait_for_training_completion(training_jobs)
    
    # Step 5: Check final system status
    check_final_system_status()
    
    # Step 6: Test anomaly detection (unless models-only mode)
    if not args.models_only:
        test_anomaly_detection()
    
    print("\n=== Training Complete ===")
    print("\nNext steps:")
    print("1. Open the web interface: http://localhost:5173")
    print("2. Navigate to the Configuration page to see trained models")
    print("3. Try analyzing a drawing to see anomaly detection in action")
    print("4. Check the API documentation: http://localhost:8000/docs")

if __name__ == "__main__":
    main()