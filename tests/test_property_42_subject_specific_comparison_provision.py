"""
Property test for subject-specific comparison provision.

**Feature: children-drawing-anomaly-detection, Property 42: Subject-Specific Comparison Provision**
**Validates: Requirements 6.7**

This test validates that the comparison service correctly provides subject-specific examples
when available and implements appropriate fallback strategies when subject-specific examples
are unavailable.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any, Optional

from app.services.comparison_service import ComparisonService


class TestSubjectSpecificComparisonProvision:
    """Test subject-specific comparison provision property."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.comparison_service = ComparisonService()
        self.mock_db = Mock()
        
        # Mock the embedding storage to avoid serialization issues
        import numpy as np
        
        mock_embedding_storage = Mock()
        mock_embedding_storage.retrieve_embedding.return_value = np.random.rand(832).astype(np.float32)
        self.comparison_service.embedding_storage = mock_embedding_storage
    
    @given(
        age_min=st.floats(min_value=2.0, max_value=10.0),
        age_max=st.floats(min_value=2.0, max_value=18.0),
        subject_category=st.sampled_from([
            "person", "house", "tree", "car", "animal", "unspecified", None
        ]),
        target_drawing_id=st.integers(min_value=1, max_value=1000),
        has_subject_examples=st.booleans(),
        has_fallback_examples=st.booleans()
    )
    @settings(max_examples=100, deadline=None)
    def test_subject_specific_comparison_provision(
        self,
        age_min: float,
        age_max: float,
        subject_category: Optional[str],
        target_drawing_id: int,
        has_subject_examples: bool,
        has_fallback_examples: bool
    ):
        """
        **Property 42: Subject-Specific Comparison Provision**
        
        For any age range and subject category, when subject-specific examples are requested,
        the comparison service should either provide subject-matched examples or implement
        appropriate fallback strategies with clear indication of the fallback used.
        """
        # Ensure age_max >= age_min
        assume(age_max >= age_min)
        
        # Arrange - Mock database responses
        self._setup_mock_database_responses(
            age_min, age_max, subject_category, target_drawing_id,
            has_subject_examples, has_fallback_examples
        )
        
        # Act
        result = self.comparison_service.get_comparison_examples_with_fallback(
            target_drawing_id=target_drawing_id,
            age_group_min=age_min,
            age_group_max=age_max,
            subject_category=subject_category,
            db=self.mock_db,
            max_examples=3
        )
        
        # Assert - Basic structure validation
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "examples" in result, "Result should contain examples list"
        assert "fallback_used" in result, "Result should indicate if fallback was used"
        assert "subject_requested" in result, "Result should track requested subject"
        assert "subject_matched" in result, "Result should track matched subject"
        
        # Assert - Subject tracking validation
        assert result["subject_requested"] == subject_category, \
            "Requested subject should be preserved"
        
        # Assert - Fallback logic validation
        if subject_category and subject_category != "unspecified":
            if has_subject_examples:
                # Should find subject-specific examples without fallback
                assert not result["fallback_used"], \
                    "Should not use fallback when subject-specific examples are available"
                assert result["subject_matched"] == subject_category, \
                    "Should match the requested subject when examples are available"
                assert len(result["examples"]) > 0, \
                    "Should return examples when subject-specific examples exist"
            else:
                # Should use fallback strategy
                if has_fallback_examples:
                    assert result["fallback_used"], \
                        "Should use fallback when no subject-specific examples available"
                    assert result["subject_matched"] == "any", \
                        "Should indicate fallback to any subject when using fallback"
                    assert "fallback_reason" in result, \
                        "Should provide reason for fallback"
                    assert len(result["examples"]) > 0, \
                        "Should return fallback examples when available"
                else:
                    # No examples available at all
                    assert result["fallback_used"], \
                        "Should indicate fallback was attempted"
                    assert len(result["examples"]) == 0, \
                        "Should return empty list when no examples available"
        else:
            # No specific subject requested or unspecified
            if has_fallback_examples:
                assert len(result["examples"]) > 0, \
                    "Should return examples when available for any subject"
                # Fallback may or may not be used depending on implementation
            else:
                assert len(result["examples"]) == 0, \
                    "Should return empty list when no examples available"
        
        # Assert - Example structure validation
        for example in result["examples"]:
            assert isinstance(example, dict), "Each example should be a dictionary"
            assert "drawing_id" in example, "Example should have drawing_id"
            assert "similarity_score" in example, "Example should have similarity_score"
            assert "drawing_info" in example, "Example should have drawing_info"
            
            # Validate drawing info structure
            drawing_info = example["drawing_info"]
            assert isinstance(drawing_info, dict), "Drawing info should be a dictionary"
            assert "age_years" in drawing_info, "Drawing info should have age"
            assert "subject" in drawing_info, "Drawing info should have subject"
            
            # Validate age is within range
            age = drawing_info["age_years"]
            assert age_min <= age <= age_max, \
                f"Example age {age} should be within range [{age_min}, {age_max}]"
            
            # Validate subject matching when not using fallback
            if not result["fallback_used"] and subject_category and subject_category != "unspecified":
                example_subject = drawing_info["subject"]
                assert example_subject == subject_category, \
                    f"Example subject {example_subject} should match requested {subject_category}"
    
    def _setup_mock_database_responses(
        self,
        age_min: float,
        age_max: float,
        subject_category: Optional[str],
        target_drawing_id: int,
        has_subject_examples: bool,
        has_fallback_examples: bool
    ):
        """Set up mock database responses for testing."""
        # Mock the find_similar_normal_examples method directly
        # This is simpler and more reliable than mocking the complex database queries
        
        def create_mock_example(drawing_id: int, subject: str, age: float):
            return {
                "drawing_id": drawing_id,
                "similarity_score": 0.85,
                "drawing_info": {
                    "filename": f"drawing_{drawing_id}.png",
                    "age_years": age,
                    "subject": subject,
                    "file_path": f"/path/to/drawing_{drawing_id}.png",
                    "anomaly_score": 0.3,
                    "normalized_score": 0.25
                }
            }
        
        # Mock the find_similar_normal_examples method
        original_method = self.comparison_service.find_similar_normal_examples
        
        def mock_find_similar_normal_examples(
            target_drawing_id,
            age_group_min,
            age_group_max,
            db,
            max_examples=3,
            similarity_threshold=0.8,
            subject_category=None
        ):
            # Determine if this is a subject-specific call or fallback call
            if subject_category and subject_category != "unspecified":
                # Subject-specific call
                if has_subject_examples:
                    # Return subject-specific examples
                    # Use age within the valid range
                    example_age = age_min if age_min == age_max else age_min + (age_max - age_min) * 0.5
                    return [
                        create_mock_example(100 + i, subject_category, example_age)
                        for i in range(2)
                    ]
                else:
                    # No subject-specific examples available
                    return []
            else:
                # Fallback call (no subject filter)
                if has_fallback_examples:
                    # Return general examples
                    # Use age within the valid range
                    example_age = age_min if age_min == age_max else age_min + (age_max - age_min) * 0.5
                    return [
                        create_mock_example(200 + i, "person", example_age)
                        for i in range(2)
                    ]
                else:
                    # No fallback examples available
                    return []
        
        self.comparison_service.find_similar_normal_examples = mock_find_similar_normal_examples
    
    @given(
        age_min=st.floats(min_value=2.0, max_value=10.0),
        age_max=st.floats(min_value=2.0, max_value=18.0),
        subject_category=st.sampled_from(["person", "house", "tree", "car", "animal"]),
        max_examples=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=50, deadline=None)
    def test_subject_specific_examples_method(
        self,
        age_min: float,
        age_max: float,
        subject_category: str,
        max_examples: int
    ):
        """Test the get_subject_specific_examples method directly."""
        # Ensure age_max >= age_min
        assume(age_max >= age_min)
        
        # Mock database setup
        mock_query = Mock()
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        
        # Create mock normal examples
        normal_results = []
        for i in range(min(max_examples, 3)):
            mock_result = Mock()
            mock_result.id = i + 1
            mock_result.filename = f"normal_{i}.png"
            mock_result.age_years = age_min + (age_max - age_min) * 0.5
            mock_result.subject = subject_category
            mock_result.file_path = f"/path/to/normal_{i}.png"
            mock_result.anomaly_score = 0.2
            mock_result.normalized_score = 0.15
            normal_results.append(mock_result)
        
        mock_query.all.return_value = normal_results
        self.mock_db.query.return_value = mock_query
        
        # Act
        result = self.comparison_service.get_subject_specific_examples(
            age_group_min=age_min,
            age_group_max=age_max,
            subject_category=subject_category,
            db=self.mock_db,
            max_examples=max_examples,
            include_anomalous=False
        )
        
        # Assert
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "normal" in result, "Result should contain normal examples"
        assert isinstance(result["normal"], list), "Normal examples should be a list"
        assert len(result["normal"]) <= max_examples, \
            f"Should not exceed max_examples ({max_examples})"
        
        # Validate example structure
        for example in result["normal"]:
            assert example["subject"] == subject_category, \
                "All examples should match the requested subject"
            assert age_min <= example["age_years"] <= age_max, \
                "All examples should be within the age range"
            assert example["category"] == "normal", \
                "All examples should be marked as normal"
    
    def test_fallback_behavior_with_empty_results(self):
        """Test fallback behavior when no examples are found."""
        # Mock empty database results
        mock_query = Mock()
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []  # No results
        
        self.mock_db.query.return_value = mock_query
        
        # Act
        result = self.comparison_service.get_comparison_examples_with_fallback(
            target_drawing_id=1,
            age_group_min=4.0,
            age_group_max=5.0,
            subject_category="person",
            db=self.mock_db,
            max_examples=3
        )
        
        # Assert
        assert result["fallback_used"] == True, "Should indicate fallback was used"
        assert len(result["examples"]) == 0, "Should return empty examples list"
        assert "fallback_reason" in result, "Should provide fallback reason"
        assert result["subject_requested"] == "person", "Should track requested subject"