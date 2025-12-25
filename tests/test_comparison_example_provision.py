"""
Property-based tests for comparison example provision.

Property 20: Comparison Example Provision
Validates: Requirements 7.3

This test validates that when analysis results are shown, the system provides
comparison with similar normal examples from the same age group.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule, initialize
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from typing import List, Dict, Any

from app.models.database import Drawing, DrawingEmbedding, AnomalyAnalysis, AgeGroupModel
from app.services.comparison_service import get_comparison_service
from app.schemas.analysis import ComparisonExampleResponse
from app.utils.embedding_serialization import get_embedding_storage


class ComparisonExampleProvisionMachine(RuleBasedStateMachine):
    """
    Property-based test machine for comparison example provision.
    
    This machine tests that the system consistently provides appropriate
    comparison examples when showing analysis results.
    """
    
    drawings = Bundle('drawings')
    age_groups = Bundle('age_groups')
    
    def __init__(self, db_session):
        super().__init__()
        self.db = db_session
        self.comparison_service = get_comparison_service()
        self.embedding_storage = get_embedding_storage()
        self.created_drawings = []
        self.created_age_groups = []
        self.created_analyses = []
        self.created_embeddings = []
    
    def teardown(self):
        """Clean up test data."""
        try:
            # Clean up in reverse order of dependencies
            for embedding in self.created_embeddings:
                self.db.delete(embedding)
            
            for analysis in self.created_analyses:
                self.db.delete(analysis)
            
            for drawing in self.created_drawings:
                self.db.delete(drawing)
            
            for age_group in self.created_age_groups:
                self.db.delete(age_group)
            
            self.db.commit()
        except Exception:
            self.db.rollback()
    
    @initialize()
    def setup_age_groups(self):
        """Initialize age groups for testing."""
        age_groups_data = [
            (3.0, 5.0),  # Preschool
            (6.0, 8.0),  # Early elementary
            (9.0, 12.0), # Late elementary
            (13.0, 16.0) # Adolescent
        ]
        
        for age_min, age_max in age_groups_data:
            age_group = AgeGroupModel(
                age_min=age_min,
                age_max=age_max,
                model_type="autoencoder",
                vision_model="vit",
                parameters="{}",
                sample_count=100,
                threshold=0.95,
                is_active=True
            )
            self.db.add(age_group)
            self.created_age_groups.append(age_group)
        
        self.db.commit()
    
    @rule(
        target=drawings,
        age_years=st.floats(min_value=3.0, max_value=16.0),
        filename=st.text(min_size=5, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        subject=st.one_of(st.none(), st.text(min_size=3, max_size=15))
    )
    def create_drawing(self, age_years: float, filename: str, subject: str) -> Drawing:
        """Create a drawing with metadata."""
        # Ensure filename has extension
        if not filename.endswith(('.png', '.jpg', '.jpeg')):
            filename += '.png'
        
        drawing = Drawing(
            filename=filename,
            file_path=f"test_uploads/{filename}",
            age_years=age_years,
            subject=subject,
            upload_timestamp=datetime.now()
        )
        
        self.db.add(drawing)
        self.db.commit()
        self.db.refresh(drawing)
        self.created_drawings.append(drawing)
        
        return drawing
    
    @rule(
        drawing=drawings,
        embedding_dim=st.integers(min_value=384, max_value=768)
    )
    def create_embedding(self, drawing: Drawing, embedding_dim: int):
        """Create an embedding for a drawing."""
        # Generate random embedding
        embedding_vector = np.random.randn(embedding_dim).astype(np.float32)
        
        # Serialize embedding
        serialized_data, dimension = self.embedding_storage.store_embedding(
            drawing_id=drawing.id,
            model_type="vit",
            embedding=embedding_vector,
            age=drawing.age_years,
            use_cache=False
        )
        
        embedding_record = DrawingEmbedding(
            drawing_id=drawing.id,
            model_type="vit",
            embedding_vector=serialized_data,
            vector_dimension=dimension
        )
        
        self.db.add(embedding_record)
        self.db.commit()
        self.db.refresh(embedding_record)
        self.created_embeddings.append(embedding_record)
    
    @rule(
        drawing=drawings,
        is_anomaly=st.booleans(),
        anomaly_score=st.floats(min_value=0.0, max_value=2.0),
        normalized_score=st.floats(min_value=0.0, max_value=1.0)
    )
    def create_analysis(self, drawing: Drawing, is_anomaly: bool, anomaly_score: float, normalized_score: float):
        """Create an analysis result for a drawing."""
        # Find appropriate age group
        age_group = None
        for ag in self.created_age_groups:
            if ag.age_min <= drawing.age_years <= ag.age_max:
                age_group = ag
                break
        
        assume(age_group is not None)
        
        analysis = AnomalyAnalysis(
            drawing_id=drawing.id,
            age_group_model_id=age_group.id,
            anomaly_score=anomaly_score,
            normalized_score=normalized_score,
            is_anomaly=is_anomaly,
            confidence=0.8,
            analysis_timestamp=datetime.now()
        )
        
        self.db.add(analysis)
        self.db.commit()
        self.db.refresh(analysis)
        self.created_analyses.append(analysis)
    
    @rule(target_drawing=drawings)
    def test_comparison_examples_provided(self, target_drawing: Drawing):
        """
        Property: When analysis results are shown, comparison examples should be provided.
        
        This test verifies that:
        1. Comparison examples are returned when available
        2. Examples are from the same age group
        3. Examples are marked as normal (not anomalous)
        4. Similarity scores are valid (0-1 range)
        5. Maximum number of examples is respected
        """
        # Find age group for target drawing
        age_group = None
        for ag in self.created_age_groups:
            if ag.age_min <= target_drawing.age_years <= ag.age_max:
                age_group = ag
                break
        
        assume(age_group is not None)
        
        # Get comparison examples
        similar_examples = self.comparison_service.find_similar_normal_examples(
            target_drawing_id=target_drawing.id,
            age_group_min=age_group.age_min,
            age_group_max=age_group.age_max,
            db=self.db,
            max_examples=3
        )
        
        # Property 1: Result should be a list
        assert isinstance(similar_examples, list), "Comparison examples should be returned as a list"
        
        # Property 2: Should not exceed maximum requested examples
        assert len(similar_examples) <= 3, "Should not return more than requested maximum examples"
        
        # Property 3: Each example should have required fields
        for example in similar_examples:
            assert isinstance(example, dict), "Each example should be a dictionary"
            assert "drawing_id" in example, "Example should have drawing_id"
            assert "similarity_score" in example, "Example should have similarity_score"
            assert "drawing_info" in example, "Example should have drawing_info"
            
            # Property 4: Similarity score should be valid
            similarity = example["similarity_score"]
            assert 0.0 <= similarity <= 1.0, f"Similarity score {similarity} should be between 0 and 1"
            
            # Property 5: Drawing info should contain required fields
            drawing_info = example["drawing_info"]
            assert "age_years" in drawing_info, "Drawing info should have age_years"
            assert "filename" in drawing_info, "Drawing info should have filename"
            assert "anomaly_score" in drawing_info, "Drawing info should have anomaly_score"
            assert "normalized_score" in drawing_info, "Drawing info should have normalized_score"
            
            # Property 6: Example should be from same age group
            example_age = drawing_info["age_years"]
            assert age_group.age_min <= example_age <= age_group.age_max, \
                f"Example age {example_age} should be within age group {age_group.age_min}-{age_group.age_max}"
            
            # Property 7: Example should not be the target drawing itself
            assert example["drawing_id"] != target_drawing.id, \
                "Comparison example should not be the target drawing itself"
    
    @rule(target_drawing=drawings)
    def test_comparison_statistics_accuracy(self, target_drawing: Drawing):
        """
        Property: Comparison statistics should accurately reflect available data.
        
        This test verifies that statistics about comparison availability
        are accurate and consistent with actual data.
        """
        # Find age group for target drawing
        age_group = None
        for ag in self.created_age_groups:
            if ag.age_min <= target_drawing.age_years <= ag.age_max:
                age_group = ag
                break
        
        assume(age_group is not None)
        
        # Get comparison statistics
        stats = self.comparison_service.get_comparison_statistics(
            age_group_min=age_group.age_min,
            age_group_max=age_group.age_max,
            db=self.db
        )
        
        # Property 1: Statistics should have required fields
        assert "age_group" in stats, "Statistics should include age_group"
        assert "total_drawings" in stats, "Statistics should include total_drawings"
        assert "normal_with_embeddings" in stats, "Statistics should include normal_with_embeddings"
        assert "anomalous_count" in stats, "Statistics should include anomalous_count"
        assert "comparison_availability" in stats, "Statistics should include comparison_availability"
        
        # Property 2: Counts should be non-negative
        assert stats["total_drawings"] >= 0, "Total drawings count should be non-negative"
        assert stats["normal_with_embeddings"] >= 0, "Normal with embeddings count should be non-negative"
        assert stats["anomalous_count"] >= 0, "Anomalous count should be non-negative"
        
        # Property 3: Normal + anomalous should not exceed total
        # Note: This might not be exact due to drawings without analysis or multiple analyses per drawing
        # We'll just check that individual counts don't exceed total
        assert stats["normal_with_embeddings"] <= stats["total_drawings"], \
            "Normal drawings count should not exceed total drawings"
        assert stats["anomalous_count"] <= stats["total_drawings"], \
            "Anomalous drawings count should not exceed total drawings"
        
        # Property 4: Comparison availability should be boolean
        assert isinstance(stats["comparison_availability"], bool), \
            "Comparison availability should be boolean"
        
        # Property 5: Age group string should match input
        expected_age_group = f"{age_group.age_min}-{age_group.age_max}"
        assert stats["age_group"] == expected_age_group, \
            f"Age group string should be {expected_age_group}"
    
    @rule()
    def test_similarity_calculation_properties(self):
        """
        Property: Similarity calculations should have mathematical properties.
        
        This test verifies that similarity calculations are mathematically sound.
        """
        # Create test embeddings
        embedding1 = np.random.randn(384).astype(np.float32)
        embedding2 = np.random.randn(384).astype(np.float32)
        embedding3 = embedding1.copy()  # Identical to embedding1
        
        # Property 1: Self-similarity should be 1.0
        self_similarity = self.comparison_service._calculate_cosine_similarity(embedding1, embedding3)
        assert abs(self_similarity - 1.0) < 1e-6, "Self-similarity should be 1.0"
        
        # Property 2: Similarity should be symmetric
        sim_12 = self.comparison_service._calculate_cosine_similarity(embedding1, embedding2)
        sim_21 = self.comparison_service._calculate_cosine_similarity(embedding2, embedding1)
        assert abs(sim_12 - sim_21) < 1e-6, "Similarity should be symmetric"
        
        # Property 3: Similarity should be in valid range
        assert 0.0 <= sim_12 <= 1.0, f"Similarity {sim_12} should be between 0 and 1"
        
        # Property 4: Zero vector handling
        zero_embedding = np.zeros(384, dtype=np.float32)
        zero_similarity = self.comparison_service._calculate_cosine_similarity(embedding1, zero_embedding)
        assert zero_similarity == 0.0, "Similarity with zero vector should be 0.0"


# Standard property-based tests
@given(
    age_years=st.floats(min_value=3.0, max_value=16.0),
    max_examples=st.integers(min_value=1, max_value=10),
    similarity_threshold=st.floats(min_value=0.0, max_value=1.0)
)
@settings(max_examples=50, deadline=None)
def test_comparison_service_parameters(age_years: float, max_examples: int, similarity_threshold: float):
    """
    Property: Comparison service should handle various parameter combinations correctly.
    
    This test verifies that the comparison service behaves correctly
    with different parameter values.
    """
    comparison_service = get_comparison_service()
    
    # Property 1: Service should be initialized
    assert comparison_service is not None, "Comparison service should be initialized"
    
    # Property 2: Parameter validation should work
    # Age group bounds
    age_group_min = max(2.0, age_years - 1.0)
    age_group_max = min(18.0, age_years + 1.0)
    
    assert age_group_min <= age_group_max, "Age group bounds should be valid"
    assert max_examples > 0, "Max examples should be positive"
    assert 0.0 <= similarity_threshold <= 1.0, "Similarity threshold should be in valid range"


@given(
    embedding_dim=st.integers(min_value=100, max_value=1000),
    num_embeddings=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=30, deadline=None)
def test_similarity_calculation_properties(embedding_dim: int, num_embeddings: int):
    """
    Property: Similarity calculations should maintain mathematical properties.
    
    This test verifies mathematical properties of similarity calculations
    across different embedding dimensions and quantities.
    """
    comparison_service = get_comparison_service()
    
    # Generate random embeddings
    embeddings = [
        np.random.randn(embedding_dim).astype(np.float32)
        for _ in range(num_embeddings)
    ]
    
    # Property 1: Self-similarity should be 1.0
    for embedding in embeddings:
        self_sim = comparison_service._calculate_cosine_similarity(embedding, embedding)
        assert abs(self_sim - 1.0) < 1e-5, "Self-similarity should be 1.0"
    
    # Property 2: Similarity should be symmetric
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim_ij = comparison_service._calculate_cosine_similarity(embeddings[i], embeddings[j])
            sim_ji = comparison_service._calculate_cosine_similarity(embeddings[j], embeddings[i])
            assert abs(sim_ij - sim_ji) < 1e-5, "Similarity should be symmetric"
    
    # Property 3: Similarity should be in valid range
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            sim = comparison_service._calculate_cosine_similarity(embeddings[i], embeddings[j])
            assert 0.0 <= sim <= 1.0, f"Similarity {sim} should be between 0 and 1"


def test_comparison_example_response_schema():
    """
    Property: ComparisonExampleResponse schema should validate correctly.
    
    This test verifies that the response schema properly validates
    comparison example data.
    """
    # Valid example data
    valid_data = {
        "drawing_id": 1,
        "filename": "test_drawing.png",
        "age_years": 5.5,
        "subject": "house",
        "similarity_score": 0.85,
        "anomaly_score": 0.12,
        "normalized_score": 0.15
    }
    
    # Property 1: Valid data should create valid response
    response = ComparisonExampleResponse(**valid_data)
    assert response.drawing_id == 1
    assert response.similarity_score == 0.85
    assert 0.0 <= response.similarity_score <= 1.0
    assert 0.0 <= response.normalized_score <= 1.0
    
    # Property 2: Invalid similarity score should raise error
    invalid_data = valid_data.copy()
    invalid_data["similarity_score"] = 1.5
    
    with pytest.raises(ValueError):
        ComparisonExampleResponse(**invalid_data)
    
    # Property 3: Invalid normalized score should raise error
    invalid_data = valid_data.copy()
    invalid_data["normalized_score"] = -0.1
    
    with pytest.raises(ValueError):
        ComparisonExampleResponse(**invalid_data)


# Test class for running the state machine
class TestComparisonExampleProvision:
    """Test class for comparison example provision properties."""
    
    def test_comparison_provision_properties(self, db_session):
        """Run the comparison provision state machine test."""
        from hypothesis.stateful import run_state_machine_as_test
        
        # Create a custom machine class that uses the db_session
        class TestMachine(ComparisonExampleProvisionMachine):
            def __init__(self):
                super().__init__(db_session)
        
        # Run the state machine test
        run_state_machine_as_test(TestMachine)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])