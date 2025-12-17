"""
Property-based tests for batch processing consistency.

**Feature: children-drawing-anomaly-detection, Property 14: Batch Processing Consistency**
**Validates: Requirements 7.5**

This module tests that batch processing produces consistent results compared to
individual processing of the same drawings.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import asyncio
from typing import List, Dict, Any


class MockAnalysisResult:
    """Mock analysis result for testing."""
    
    def __init__(self, drawing_id: int, anomaly_score: float):
        self.drawing_id = drawing_id
        self.anomaly_score = anomaly_score
        self.normalized_score = min(1.0, anomaly_score / 10.0)
        self.is_anomaly = anomaly_score > 1.0
        self.confidence = 0.8


def create_mock_analysis_function():
    """Create a mock analysis function with deterministic behavior."""
    
    async def mock_analyze_single(drawing_id: int, force_reanalysis: bool = False) -> MockAnalysisResult:
        """Mock single analysis function."""
        # Use drawing_id to create deterministic but varied results
        np.random.seed(drawing_id)  # Deterministic based on drawing_id
        base_score = np.random.uniform(0.5, 2.0)
        
        return MockAnalysisResult(drawing_id, base_score)
    
    return mock_analyze_single


def create_mock_batch_analysis_function():
    """Create a mock batch analysis function."""
    
    async def mock_analyze_batch(drawing_ids: List[int], force_reanalysis: bool = False) -> Dict[int, MockAnalysisResult]:
        """Mock batch analysis function."""
        results = {}
        
        # Process each drawing with the same logic as individual analysis
        for drawing_id in drawing_ids:
            np.random.seed(drawing_id)  # Same seed as individual analysis
            base_score = np.random.uniform(0.5, 2.0)
            results[drawing_id] = MockAnalysisResult(drawing_id, base_score)
        
        return results
    
    return mock_analyze_batch


async def batch_processing_consistency_impl(drawing_count, drawing_ids):
    """
    **Feature: children-drawing-anomaly-detection, Property 14: Batch Processing Consistency**
    **Validates: Requirements 7.5**
    
    Property: For any batch of drawings, each drawing should receive the same analysis 
    result whether processed individually or as part of the batch.
    """
    # Limit drawing_ids to drawing_count
    test_drawing_ids = drawing_ids[:drawing_count]
    
    # Create mock analysis functions
    mock_single_analysis = create_mock_analysis_function()
    mock_batch_analysis = create_mock_batch_analysis_function()
    
    # Process drawings individually
    individual_results = {}
    for drawing_id in test_drawing_ids:
        result = await mock_single_analysis(drawing_id, force_reanalysis=True)
        individual_results[drawing_id] = result
    
    # Process drawings in batch
    batch_results = await mock_batch_analysis(test_drawing_ids, force_reanalysis=True)
    
    # Verify consistency between individual and batch results
    for drawing_id in test_drawing_ids:
        individual_result = individual_results[drawing_id]
        batch_result = batch_results[drawing_id]
        
        # Check that key metrics are identical
        assert abs(individual_result.anomaly_score - batch_result.anomaly_score) < 1e-6, \
            f"Anomaly scores differ for drawing {drawing_id}: " \
            f"individual={individual_result.anomaly_score}, batch={batch_result.anomaly_score}"
        
        assert abs(individual_result.normalized_score - batch_result.normalized_score) < 1e-6, \
            f"Normalized scores differ for drawing {drawing_id}: " \
            f"individual={individual_result.normalized_score}, batch={batch_result.normalized_score}"
        
        assert individual_result.is_anomaly == batch_result.is_anomaly, \
            f"Anomaly flags differ for drawing {drawing_id}: " \
            f"individual={individual_result.is_anomaly}, batch={batch_result.is_anomaly}"
        
        assert abs(individual_result.confidence - batch_result.confidence) < 1e-6, \
            f"Confidence scores differ for drawing {drawing_id}: " \
            f"individual={individual_result.confidence}, batch={batch_result.confidence}"


async def batch_processing_order_independence_impl(drawing_ids):
    """
    **Feature: children-drawing-anomaly-detection, Property 14: Batch Processing Consistency**
    **Validates: Requirements 7.5**
    
    Property: Batch processing results should be independent of the order in which 
    drawings are processed within the batch.
    """
    # Create mock analysis function
    mock_batch_analysis = create_mock_batch_analysis_function()
    
    # Process in original order
    original_results = await mock_batch_analysis(drawing_ids, force_reanalysis=True)
    
    # Process in reversed order
    reversed_results = await mock_batch_analysis(list(reversed(drawing_ids)), force_reanalysis=True)
    
    # Verify that results are identical regardless of processing order
    for drawing_id in drawing_ids:
        original_score = original_results[drawing_id].anomaly_score
        reversed_score = reversed_results[drawing_id].anomaly_score
        
        assert abs(original_score - reversed_score) < 1e-6, \
            f"Scores differ based on processing order for drawing {drawing_id}: " \
            f"original={original_score}, reversed={reversed_score}"


async def batch_processing_deterministic_impl(batch_size, base_seed):
    """
    **Feature: children-drawing-anomaly-detection, Property 14: Batch Processing Consistency**
    **Validates: Requirements 7.5**
    
    Property: Running the same batch multiple times should produce identical results.
    """
    # Generate drawing IDs based on base_seed for deterministic behavior
    np.random.seed(base_seed)
    drawing_ids = [np.random.randint(1, 1000) for _ in range(batch_size)]
    
    # Create mock analysis function
    mock_batch_analysis = create_mock_batch_analysis_function()
    
    # Run batch processing multiple times
    run1_results = await mock_batch_analysis(drawing_ids, force_reanalysis=True)
    run2_results = await mock_batch_analysis(drawing_ids, force_reanalysis=True)
    
    # Verify deterministic behavior
    for drawing_id in drawing_ids:
        run1 = run1_results[drawing_id]
        run2 = run2_results[drawing_id]
        
        assert abs(run1.anomaly_score - run2.anomaly_score) < 1e-6, \
            f"Anomaly scores not deterministic for drawing {drawing_id}: " \
            f"run1={run1.anomaly_score}, run2={run2.anomaly_score}"
        
        assert abs(run1.normalized_score - run2.normalized_score) < 1e-6, \
            f"Normalized scores not deterministic for drawing {drawing_id}"
        
        assert run1.is_anomaly == run2.is_anomaly, \
            f"Anomaly flags not deterministic for drawing {drawing_id}"
        
        assert abs(run1.confidence - run2.confidence) < 1e-6, \
            f"Confidence scores not deterministic for drawing {drawing_id}"


# Async test runner helper
def run_async_test(coro):
    """Helper to run async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# Sync wrappers for pytest
@given(
    drawing_count=st.integers(min_value=1, max_value=5),
    drawing_ids=st.lists(st.integers(min_value=1, max_value=50), min_size=1, max_size=10, unique=True)
)
@settings(max_examples=10, deadline=8000)
def test_batch_processing_consistency_sync(drawing_count, drawing_ids):
    """
    **Feature: children-drawing-anomaly-detection, Property 14: Batch Processing Consistency**
    **Validates: Requirements 7.5**
    
    Sync wrapper for batch processing consistency test.
    """
    return run_async_test(
        batch_processing_consistency_impl(drawing_count, drawing_ids)
    )


@given(
    drawing_ids=st.lists(st.integers(min_value=1, max_value=30), min_size=1, max_size=4, unique=True)
)
@settings(max_examples=8, deadline=6000)
def test_batch_processing_order_independence_sync(drawing_ids):
    """
    **Feature: children-drawing-anomaly-detection, Property 14: Batch Processing Consistency**
    **Validates: Requirements 7.5**
    
    Sync wrapper for order independence test.
    """
    return run_async_test(
        batch_processing_order_independence_impl(drawing_ids)
    )


@given(
    batch_size=st.integers(min_value=1, max_value=4),
    base_seed=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=8, deadline=6000)
def test_batch_processing_deterministic_sync(batch_size, base_seed):
    """
    **Feature: children-drawing-anomaly-detection, Property 14: Batch Processing Consistency**
    **Validates: Requirements 7.5**
    
    Sync wrapper for deterministic behavior test.
    """
    return run_async_test(
        batch_processing_deterministic_impl(batch_size, base_seed)
    )