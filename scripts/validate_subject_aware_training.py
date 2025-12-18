#!/usr/bin/env python3
"""
Validation script for subject-aware training workflow.

This script validates that:
1. All models are subject-aware
2. Training reports include subject statistics
3. Embeddings are in hybrid format
4. Subject stratification is working correctly

Requirements: 12.2
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.database import AgeGroupModel, DrawingEmbedding, Drawing, TrainingReport
from app.utils.embedding_serialization import deserialize_embedding_from_db

def get_database_session():
    """Get database session."""
    DATABASE_URL = 'sqlite:///./drawings.db'
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()

def validate_models_are_subject_aware():
    """Validate that all models are subject-aware."""
    print("🔍 Validating subject-aware models...")
    
    with get_database_session() as db:
        models = db.query(AgeGroupModel).filter(AgeGroupModel.is_active == True).all()
        
        print(f"  Found {len(models)} active models")
        
        issues = []
        
        for model in models:
            # Check supports_subjects flag
            if not model.supports_subjects:
                issues.append(f"Model {model.id} (age {model.age_min}-{model.age_max}) does not support subjects")
            
            # Check embedding_type
            if model.embedding_type != "hybrid":
                issues.append(f"Model {model.id} embedding_type is '{model.embedding_type}', expected 'hybrid'")
            
            # Check subject_categories
            if not model.subject_categories:
                issues.append(f"Model {model.id} has no subject_categories defined")
            else:
                try:
                    categories = json.loads(model.subject_categories)
                    if not isinstance(categories, list) or len(categories) == 0:
                        issues.append(f"Model {model.id} has invalid subject_categories")
                except json.JSONDecodeError:
                    issues.append(f"Model {model.id} has malformed subject_categories JSON")
            
            # Check model parameters for subject-aware info
            try:
                params = json.loads(model.parameters)
                if "subject_distribution" not in params:
                    issues.append(f"Model {model.id} parameters missing subject_distribution")
                if params.get("embedding_type") != "hybrid":
                    issues.append(f"Model {model.id} parameters embedding_type is not 'hybrid'")
            except json.JSONDecodeError:
                issues.append(f"Model {model.id} has malformed parameters JSON")
        
        if issues:
            print("  ❌ Issues found:")
            for issue in issues:
                print(f"    - {issue}")
            return False
        else:
            print("  ✅ All models are properly configured as subject-aware")
            return True

def validate_embeddings_are_hybrid():
    """Validate that embeddings are in hybrid format."""
    print("🔍 Validating hybrid embeddings...")
    
    with get_database_session() as db:
        # Sample some embeddings to check
        sample_embeddings = db.query(DrawingEmbedding).limit(100).all()
        
        print(f"  Checking {len(sample_embeddings)} sample embeddings")
        
        issues = []
        dimension_counts = {}
        
        for embedding in sample_embeddings:
            # Check embedding_type flag
            if embedding.embedding_type != "hybrid":
                issues.append(f"Embedding {embedding.id} type is '{embedding.embedding_type}', expected 'hybrid'")
            
            # Check vector_dimension
            if embedding.vector_dimension != 832:
                issues.append(f"Embedding {embedding.id} dimension is {embedding.vector_dimension}, expected 832")
            
            # Check actual vector dimensions
            try:
                vector = deserialize_embedding_from_db(embedding.embedding_vector)
                actual_dim = len(vector)
                dimension_counts[actual_dim] = dimension_counts.get(actual_dim, 0) + 1
                
                if actual_dim != 832:
                    issues.append(f"Embedding {embedding.id} actual vector is {actual_dim}-dim, expected 832")
            except Exception as e:
                issues.append(f"Embedding {embedding.id} deserialization failed: {e}")
            
            # Check component separation
            if embedding.visual_component is None:
                issues.append(f"Embedding {embedding.id} missing visual_component")
            if embedding.subject_component is None:
                issues.append(f"Embedding {embedding.id} missing subject_component")
        
        print(f"  📊 Dimension distribution: {dimension_counts}")
        
        if issues:
            print("  ❌ Issues found:")
            for issue in issues[:10]:  # Show first 10 issues
                print(f"    - {issue}")
            if len(issues) > 10:
                print(f"    ... and {len(issues) - 10} more issues")
            return False
        else:
            print("  ✅ All sample embeddings are properly formatted as hybrid")
            return True

def validate_subject_stratification():
    """Validate subject stratification in the data."""
    print("🔍 Validating subject stratification...")
    
    with get_database_session() as db:
        # Get all drawings with subjects
        drawings = db.query(Drawing).all()
        
        subject_counts = {}
        age_subject_combinations = {}
        
        for drawing in drawings:
            subject = drawing.subject if drawing.subject else "unspecified"
            age_group = f"{int(drawing.age_years)}-{int(drawing.age_years)+1}"
            
            subject_counts[subject] = subject_counts.get(subject, 0) + 1
            
            key = (age_group, subject)
            age_subject_combinations[key] = age_subject_combinations.get(key, 0) + 1
        
        print(f"  📊 Total drawings: {len(drawings)}")
        print(f"  🎨 Unique subjects: {len(subject_counts)}")
        print(f"  🔗 Age-subject combinations: {len(age_subject_combinations)}")
        
        # Show top subjects
        print(f"  🏆 Top 10 subjects:")
        for subject, count in sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / len(drawings)) * 100
            print(f"    {subject}: {count} ({percentage:.1f}%)")
        
        # Check for sufficient diversity
        issues = []
        
        if len(subject_counts) < 5:
            issues.append(f"Low subject diversity: only {len(subject_counts)} categories")
        
        # Check for age-subject combinations with very few samples
        sparse_combinations = [(k, v) for k, v in age_subject_combinations.items() if v < 5]
        if len(sparse_combinations) > len(age_subject_combinations) * 0.5:
            issues.append(f"Many sparse age-subject combinations: {len(sparse_combinations)}/{len(age_subject_combinations)}")
        
        if issues:
            print("  ⚠️ Stratification concerns:")
            for issue in issues:
                print(f"    - {issue}")
            return True  # Not a failure, just a concern
        else:
            print("  ✅ Good subject stratification across age groups")
            return True

def validate_training_reports():
    """Validate that training reports include subject statistics."""
    print("🔍 Validating training reports...")
    
    with get_database_session() as db:
        reports = db.query(TrainingReport).all()
        
        print(f"  Found {len(reports)} training reports")
        
        if len(reports) == 0:
            print("  ⚠️ No training reports found - may need to retrain models")
            return True
        
        issues = []
        
        for report in reports:
            try:
                metrics = json.loads(report.metrics_summary)
                
                # Check for subject-related metrics
                expected_keys = ["subject_distribution", "embedding_type", "input_dimension"]
                missing_keys = [key for key in expected_keys if key not in metrics]
                
                if missing_keys:
                    issues.append(f"Report {report.id} missing keys: {missing_keys}")
                
                # Validate specific values
                if metrics.get("embedding_type") != "hybrid":
                    issues.append(f"Report {report.id} embedding_type is not 'hybrid'")
                
                if metrics.get("input_dimension") != 832:
                    issues.append(f"Report {report.id} input_dimension is not 832")
                
            except json.JSONDecodeError:
                issues.append(f"Report {report.id} has malformed metrics_summary JSON")
        
        if issues:
            print("  ❌ Issues found:")
            for issue in issues:
                print(f"    - {issue}")
            return False
        else:
            print("  ✅ All training reports include subject-aware statistics")
            return True

def main():
    """Run all validation checks."""
    print("=== Subject-Aware Training Validation ===\n")
    
    all_passed = True
    
    # Run all validation checks
    checks = [
        ("Models are subject-aware", validate_models_are_subject_aware),
        ("Embeddings are hybrid format", validate_embeddings_are_hybrid),
        ("Subject stratification", validate_subject_stratification),
        ("Training reports include subject stats", validate_training_reports),
    ]
    
    results = []
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        try:
            result = check_func()
            results.append((check_name, result))
            if not result:
                all_passed = False
        except Exception as e:
            print(f"  ❌ Check failed with error: {e}")
            results.append((check_name, False))
            all_passed = False
    
    # Summary
    print(f"\n=== Validation Summary ===")
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {check_name}")
    
    if all_passed:
        print(f"\n🎉 All validation checks passed!")
        print(f"✅ Subject-aware training workflow is properly configured")
    else:
        print(f"\n⚠️ Some validation checks failed")
        print(f"❌ Please review the issues above and retrain models if needed")
        sys.exit(1)

if __name__ == "__main__":
    main()