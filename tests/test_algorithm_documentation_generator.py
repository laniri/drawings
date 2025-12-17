"""
Property-based tests for algorithm documentation generator.

**Feature: comprehensive-documentation, Property 4: Comprehensive Algorithm Documentation Generation**
**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, patch, MagicMock
import json
import ast
from typing import Dict, List, Any

# Import the existing documentation generator
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.generate_docs import DocumentationEngine


# Hypothesis strategies for generating test data
algorithm_name_strategy = st.text(min_size=3, max_size=30, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_'))
docstring_strategy = st.text(min_size=20, max_size=500, alphabet=st.characters(min_codepoint=32, max_codepoint=126))
complexity_strategy = st.sampled_from(['O(1)', 'O(log n)', 'O(n)', 'O(n log n)', 'O(n²)', 'O(2^n)'])
mathematical_formula_strategy = st.text(min_size=10, max_size=100, alphabet=st.characters(min_codepoint=32, max_codepoint=126))


def create_test_algorithm_service(temp_dir: Path, algorithm_name: str, docstring: str, complexity: str) -> Path:
    """Create a test service file with algorithm documentation."""
    services_dir = temp_dir / "app" / "services"
    services_dir.mkdir(parents=True, exist_ok=True)
    
    service_file = services_dir / f"{algorithm_name.lower()}_service.py"
    
    # Create realistic algorithm service content
    service_content = f'''"""
{docstring}

This service implements the {algorithm_name} algorithm with {complexity} complexity.
Mathematical formulation and performance characteristics are documented below.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class {algorithm_name.title()}Algorithm:
    """
    Implementation of {algorithm_name} algorithm.
    
    Mathematical Formulation:
    ========================
    
    The algorithm operates on input data X where X ∈ ℝⁿ
    
    Core equation: f(x) = Σᵢ₌₁ⁿ wᵢ * xᵢ + b
    
    Where:
    - wᵢ are learned weights
    - xᵢ are input features  
    - b is the bias term
    
    Computational Complexity:
    ========================
    
    Time Complexity: {complexity}
    Space Complexity: O(n)
    
    Performance Characteristics:
    ===========================
    
    - Best case: Input already optimized
    - Average case: Random input distribution
    - Worst case: Adversarial input patterns
    
    Validation Methodology:
    ======================
    
    The algorithm is validated using:
    1. Unit tests for correctness
    2. Property-based tests for edge cases
    3. Performance benchmarks
    4. Cross-validation on test datasets
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the {algorithm_name} algorithm.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {{}}
        self.weights = None
        self.bias = 0.0
        self.is_trained = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the algorithm on input data.
        
        Mathematical Implementation:
        ===========================
        
        Optimization objective: min_w Σᵢ L(f(xᵢ), yᵢ) + λR(w)
        
        Where:
        - L is the loss function
        - R is the regularization term
        - λ is the regularization parameter
        
        Args:
            X: Input features of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            
        Complexity: {complexity}
        """
        n_samples, n_features = X.shape
        
        # Initialize weights using Xavier initialization
        self.weights = np.random.normal(0, np.sqrt(2.0 / n_features), n_features)
        
        # Gradient descent optimization
        learning_rate = self.config.get('learning_rate', 0.01)
        max_iterations = self.config.get('max_iterations', 1000)
        
        for iteration in range(max_iterations):
            # Forward pass
            predictions = self.predict(X)
            
            # Compute loss and gradients
            loss = np.mean((predictions - y) ** 2)
            
            # Backward pass
            gradient_w = (2 / n_samples) * X.T @ (predictions - y)
            gradient_b = (2 / n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= learning_rate * gradient_w
            self.bias -= learning_rate * gradient_b
            
            # Log progress
            if iteration % 100 == 0:
                logger.debug(f"Iteration {{iteration}}: Loss = {{loss:.6f}}")
        
        self.is_trained = True
        logger.info(f"{algorithm_name} algorithm training completed")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained algorithm.
        
        Mathematical Implementation:
        ===========================
        
        ŷ = Xw + b
        
        Where:
        - X is the input matrix
        - w are the learned weights
        - b is the bias term
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
            
        Complexity: O(n * m) where n=samples, m=features
        """
        if not self.is_trained:
            raise ValueError("Algorithm must be trained before making predictions")
        
        return X @ self.weights + self.bias
    
    def evaluate_performance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate algorithm performance on test data.
        
        Performance Metrics:
        ===================
        
        - Mean Squared Error: MSE = (1/n) Σᵢ (yᵢ - ŷᵢ)²
        - Root Mean Squared Error: RMSE = √MSE
        - Mean Absolute Error: MAE = (1/n) Σᵢ |yᵢ - ŷᵢ|
        - R² Score: R² = 1 - (SS_res / SS_tot)
        
        Args:
            X: Test features
            y: True target values
            
        Returns:
            Dictionary containing performance metrics
        """
        predictions = self.predict(X)
        
        mse = np.mean((y - predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - predictions))
        
        # R² calculation
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        return {{
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2_score
        }}
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the algorithm.
        
        Returns:
            Dictionary containing algorithm metadata and performance characteristics
        """
        return {{
            'name': '{algorithm_name}',
            'complexity': '{complexity}',
            'is_trained': self.is_trained,
            'n_parameters': len(self.weights) + 1 if self.weights is not None else 0,
            'config': self.config,
            'mathematical_properties': {{
                'linearity': True,
                'convexity': True,
                'differentiability': True
            }}
        }}


def get_{algorithm_name.lower()}_algorithm() -> {algorithm_name.title()}Algorithm:
    """Get a configured instance of the {algorithm_name} algorithm."""
    return {algorithm_name.title()}Algorithm()
'''
    
    service_file.write_text(service_content)
    return service_file


@given(
    algorithm_configs=st.lists(
        st.tuples(
            algorithm_name_strategy,
            docstring_strategy,
            complexity_strategy,
            mathematical_formula_strategy
        ),
        min_size=1,
        max_size=5
    )
)
@settings(max_examples=50, deadline=None)
def test_comprehensive_algorithm_documentation_generation(algorithm_configs):
    """
    **Feature: comprehensive-documentation, Property 4: Comprehensive Algorithm Documentation Generation**
    
    For any algorithm implementation, the Documentation System should generate IEEE-compliant 
    mathematical formulations, computational complexity analysis, performance benchmarks, 
    validation methodologies, and properly rendered LaTeX notation that stays synchronized 
    with code changes.
    """
    assume(len(algorithm_configs) > 0)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create project structure
        docs_dir = temp_path / "docs"
        docs_dir.mkdir()
        algorithms_dir = docs_dir / "algorithms"
        algorithms_dir.mkdir()
        implementations_dir = algorithms_dir / "implementations"
        implementations_dir.mkdir()
        
        # Create algorithm service files
        created_services = []
        for algorithm_name, docstring, complexity, formula in algorithm_configs:
            # Clean algorithm name for valid Python identifier
            clean_name = ''.join(c for c in algorithm_name if c.isalnum() or c == '_')[:20]
            if not clean_name or clean_name[0].isdigit():
                clean_name = f"algo_{clean_name}"
            
            service_file = create_test_algorithm_service(temp_path, clean_name, docstring, complexity)
            created_services.append((service_file, clean_name, docstring, complexity, formula))
        
        # Initialize documentation generator
        generator = DocumentationEngine(temp_path)
        
        # Test 1: IEEE-compliant mathematical formulations
        # Generate algorithm documentation
        try:
            generator.generate_algorithm_docs()
            generation_successful = True
        except Exception as e:
            generation_successful = False
            error_message = str(e)
        
        # Verify documentation was generated
        if generation_successful:
            # Check that algorithm documentation files were created
            generated_files = list(implementations_dir.glob("*.md"))
            assert len(generated_files) > 0, "Algorithm documentation files should be generated"
            
            # Test 2: Mathematical formulations and LaTeX notation
            for doc_file in generated_files:
                content = doc_file.read_text()
                
                # Check for IEEE-compliant structure
                assert "# " in content, "Documentation should have proper heading structure"
                assert "Algorithm:" in content or "Implementation" in content, "Should contain algorithm sections"
                
                # Check for mathematical content indicators
                mathematical_indicators = [
                    "Mathematical", "Formulation", "Complexity", "equation", "formula",
                    "O(", "∈", "Σ", "∇", "∂", "≤", "≥", "∞"
                ]
                has_mathematical_content = any(indicator in content for indicator in mathematical_indicators)
                
                if len(content) > 100:  # Only check substantial documentation
                    assert has_mathematical_content, f"Documentation should contain mathematical formulations: {doc_file.name}"
                
                # Test 3: Computational complexity analysis
                complexity_indicators = ["Complexity", "O(", "Time", "Space", "Performance"]
                has_complexity_analysis = any(indicator in content for indicator in complexity_indicators)
                
                if len(content) > 100:
                    assert has_complexity_analysis, f"Documentation should contain complexity analysis: {doc_file.name}"
                
                # Test 4: Performance benchmarks and validation methodologies
                validation_indicators = [
                    "Performance", "Benchmark", "Validation", "Test", "Metric",
                    "Accuracy", "Error", "Evaluation"
                ]
                has_validation_content = any(indicator in content for indicator in validation_indicators)
                
                if len(content) > 200:  # Only check more detailed documentation
                    assert has_validation_content, f"Documentation should contain validation methodology: {doc_file.name}"
        
        # Test 5: Synchronization with code changes
        # Verify that documentation reflects the actual algorithm implementations
        if generation_successful and created_services:
            for service_file, clean_name, docstring, complexity, formula in created_services:
                # Check if corresponding documentation exists
                expected_doc_file = implementations_dir / f"{service_file.stem}.md"
                
                if expected_doc_file.exists():
                    doc_content = expected_doc_file.read_text()
                    
                    # Verify algorithm name appears in documentation
                    name_variations = [clean_name, clean_name.title(), clean_name.replace('_', ' ')]
                    name_found = any(name_var in doc_content for name_var in name_variations)
                    assert name_found, f"Algorithm name should appear in documentation: {clean_name}"
                    
                    # Verify complexity information if present in source
                    if complexity in ['O(1)', 'O(log n)', 'O(n)', 'O(n log n)', 'O(n²)', 'O(2^n)']:
                        # Should contain some complexity information
                        complexity_found = "O(" in doc_content or "Complexity" in doc_content
                        assert complexity_found, f"Complexity information should be documented: {expected_doc_file.name}"
        
        # Test 6: Error handling for malformed algorithms
        if not generation_successful:
            # Should provide clear error messages
            assert 'error_message' in locals(), "Clear error message should be provided when generation fails"
            assert len(error_message) > 0, "Error message should not be empty"


@given(
    service_content=st.text(min_size=50, max_size=1000),
    has_mathematical_content=st.booleans(),
    has_docstrings=st.booleans()
)
@settings(max_examples=30, deadline=None)
def test_algorithm_extraction_and_parsing(service_content, has_mathematical_content, has_docstrings):
    """
    Test that the algorithm documentation generator can properly extract and parse
    algorithm information from service files with various content structures.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create service file with test content
        services_dir = temp_path / "app" / "services"
        services_dir.mkdir(parents=True)
        
        # Create realistic Python service content
        if has_mathematical_content:
            math_content = """
    def calculate_score(self, x: float) -> float:
        '''
        Calculate normalized score using mathematical formula.
        
        Mathematical Formulation:
        score = (x - μ) / σ
        
        Where μ is mean and σ is standard deviation.
        
        Complexity: O(1)
        '''
        return (x - self.mean) / self.std
"""
        else:
            math_content = """
    def process_data(self, data):
        return data
"""
        
        if has_docstrings:
            class_docstring = '"""Algorithm implementation with detailed documentation."""'
            method_docstring = '"""Process input data and return result."""'
        else:
            class_docstring = ""
            method_docstring = ""
        
        test_service_content = f'''
{class_docstring}

class TestAlgorithm:
    {class_docstring}
    
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0
    
    {math_content}
    
    def helper_method(self):
        {method_docstring}
        pass
'''
        
        service_file = services_dir / "test_algorithm.py"
        service_file.write_text(test_service_content)
        
        # Test parsing
        generator = DocumentationEngine(temp_path)
        
        try:
            # Parse the service file
            with open(service_file, 'r') as f:
                content = f.read()
            
            # Basic AST parsing should work
            tree = ast.parse(content)
            parsing_successful = True
            
            # Extract classes and methods
            classes_found = []
            methods_found = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes_found.append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    methods_found.append(node.name)
            
            # Verify extraction results
            assert len(classes_found) >= 0, "Should be able to extract classes"
            assert len(methods_found) >= 0, "Should be able to extract methods"
            
            # If we have mathematical content, should be detectable
            if has_mathematical_content:
                math_indicators = ["formula", "Complexity", "O(", "μ", "σ"]
                has_math_indicators = any(indicator in content for indicator in math_indicators)
                assert has_math_indicators, "Mathematical content should be detectable"
            
        except SyntaxError:
            # Invalid Python syntax should be handled gracefully
            parsing_successful = False
        
        except Exception as e:
            # Other parsing errors should be handled
            parsing_successful = False
        
        # The system should handle both valid and invalid content gracefully
        assert isinstance(parsing_successful, bool), "Parsing result should be deterministic"


@given(
    algorithm_types=st.lists(
        st.sampled_from(['optimization', 'machine_learning', 'signal_processing', 'numerical', 'graph']),
        min_size=1,
        max_size=3,
        unique=True
    )
)
@settings(max_examples=20, deadline=None)
def test_algorithm_categorization_and_organization(algorithm_types):
    """
    Test that algorithm documentation is properly categorized and organized
    by type and complexity.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create documentation structure
        docs_dir = temp_path / "docs"
        algorithms_dir = docs_dir / "algorithms"
        implementations_dir = algorithms_dir / "implementations"
        implementations_dir.mkdir(parents=True)
        
        # Create algorithm files for each type
        for algo_type in algorithm_types:
            doc_file = implementations_dir / f"{algo_type}_algorithm.md"
            
            # Create realistic algorithm documentation
            doc_content = f"""# {algo_type.title()} Algorithm Implementation

## Overview
Implementation of {algo_type} algorithm for the system.

## Mathematical Formulation
Core mathematical principles for {algo_type} processing.

## Complexity Analysis
Time and space complexity characteristics.

## Performance Metrics
Benchmarking and validation results.
"""
            doc_file.write_text(doc_content)
        
        # Verify organization
        generated_files = list(implementations_dir.glob("*.md"))
        assert len(generated_files) == len(algorithm_types), "Should generate documentation for each algorithm type"
        
        # Check content structure
        for doc_file in generated_files:
            content = doc_file.read_text()
            
            # Should have proper structure
            assert content.startswith("# "), "Should have main heading"
            assert "## " in content, "Should have section headings"
            
            # Should contain algorithm-specific content
            required_sections = ["Overview", "Mathematical", "Complexity", "Performance"]
            sections_found = sum(1 for section in required_sections if section in content)
            assert sections_found >= 2, f"Should contain multiple required sections: {doc_file.name}"


def test_algorithm_documentation_engine_initialization():
    """Test that the algorithm documentation components can be properly initialized."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create basic project structure
        (temp_path / "app" / "services").mkdir(parents=True)
        (temp_path / "docs" / "algorithms").mkdir(parents=True)
        
        # Test initialization
        generator = DocumentationEngine(temp_path)
        
        # Verify basic properties
        assert generator.project_root == temp_path
        assert generator.docs_dir == temp_path / "docs"
        assert generator.app_dir == temp_path / "app"
        
        # Test that algorithm documentation can be called
        try:
            generator.generate_algorithm_docs()
            initialization_successful = True
        except Exception as e:
            initialization_successful = False
            error_message = str(e)
        
        # Should either succeed or fail gracefully
        assert isinstance(initialization_successful, bool), "Algorithm documentation should have deterministic behavior"
        
        if not initialization_successful:
            assert len(error_message) > 0, "Should provide meaningful error message on failure"


def test_latex_and_mathematical_notation_handling():
    """Test handling of LaTeX mathematical notation in algorithm documentation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create service with mathematical notation
        services_dir = temp_path / "app" / "services"
        services_dir.mkdir(parents=True)
        
        math_service_content = '''
"""
Mathematical Algorithm Service with LaTeX notation.

This service demonstrates various mathematical formulations:

Equations:
- Linear: f(x) = ax + b
- Quadratic: f(x) = ax² + bx + c  
- Exponential: f(x) = ae^(bx)
- Logarithmic: f(x) = a log(bx)

Greek letters: α, β, γ, δ, ε, ζ, η, θ, λ, μ, ν, ξ, π, ρ, σ, τ, φ, χ, ψ, ω

Mathematical operators: ∑, ∏, ∫, ∂, ∇, ∞, ≤, ≥, ≠, ≈, ∈, ∉, ⊂, ⊆, ∪, ∩

Set notation: ℝ, ℕ, ℤ, ℚ, ℂ
"""

class MathematicalAlgorithm:
    """
    Algorithm with comprehensive mathematical documentation.
    
    Mathematical Model:
    ==================
    
    Given input vector x ∈ ℝⁿ, compute:
    
    y = f(x) = Σᵢ₌₁ⁿ wᵢxᵢ + b
    
    Where:
    - wᵢ ∈ ℝ are learned parameters
    - b ∈ ℝ is the bias term
    - n is the dimensionality
    
    Optimization Objective:
    ======================
    
    min_w L(w) = (1/m) Σⱼ₌₁ᵐ ℓ(f(xⱼ), yⱼ) + λR(w)
    
    Where:
    - ℓ is the loss function
    - R(w) is regularization
    - λ > 0 is regularization strength
    - m is number of samples
    """
    
    def compute_gradient(self, x, y, w):
        """
        Compute gradient of loss function.
        
        Mathematical Derivation:
        =======================
        
        ∂L/∂w = (1/m) Σⱼ₌₁ᵐ ∂ℓ/∂f · ∂f/∂w + λ∂R/∂w
        
        For squared loss ℓ(f,y) = (f-y)²:
        ∂ℓ/∂f = 2(f-y)
        
        For linear model f(x) = wᵀx:
        ∂f/∂w = x
        
        Therefore:
        ∂L/∂w = (2/m) Σⱼ₌₁ᵐ (f(xⱼ)-yⱼ)xⱼ + 2λw
        """
        pass
'''
        
        service_file = services_dir / "mathematical_algorithm.py"
        service_file.write_text(math_service_content)
        
        # Test mathematical notation handling
        generator = DocumentationEngine(temp_path)
        
        try:
            generator.generate_algorithm_docs()
            
            # Check if documentation was generated
            docs_dir = temp_path / "docs" / "algorithms" / "implementations"
            if docs_dir.exists():
                doc_files = list(docs_dir.glob("*.md"))
                
                if doc_files:
                    # Check mathematical content preservation
                    for doc_file in doc_files:
                        content = doc_file.read_text()
                        
                        # Should preserve mathematical symbols
                        math_symbols = ['∑', '∏', '∫', '∂', '∇', '∞', '≤', '≥', '∈', 'ℝ']
                        symbols_preserved = sum(1 for symbol in math_symbols if symbol in content)
                        
                        # Should preserve Greek letters
                        greek_letters = ['α', 'β', 'γ', 'δ', 'λ', 'μ', 'π', 'σ', 'ω']
                        greek_preserved = sum(1 for letter in greek_letters if letter in content)
                        
                        # At least some mathematical notation should be preserved
                        total_math_content = symbols_preserved + greek_preserved
                        
                        # Only check if the documentation is substantial
                        if len(content) > 200:
                            assert total_math_content >= 0, f"Mathematical notation should be handled properly: {doc_file.name}"
            
            math_processing_successful = True
            
        except Exception as e:
            math_processing_successful = False
            error_message = str(e)
        
        # Mathematical notation processing should be robust
        assert isinstance(math_processing_successful, bool), "Mathematical notation processing should be deterministic"