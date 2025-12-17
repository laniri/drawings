#!/usr/bin/env python3
"""
Performance Analysis Generator

This module generates performance analysis and validation specifications
for algorithm documentation.
"""

import ast
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .algorithm_documentation_generator import Algorithm, PerformanceDoc, ValidationDoc

logger = logging.getLogger(__name__)


class PerformanceAnalysisGenerator:
    """Generator for algorithm performance analysis documentation."""
    
    def __init__(self):
        self.benchmark_patterns = [
            r'benchmark', r'performance', r'speed', r'timing',
            r'throughput', r'latency', r'efficiency', r'optimization'
        ]
        
        self.complexity_patterns = [
            r'O\([^)]+\)', r'complexity', r'scalability',
            r'time complexity', r'space complexity'
        ]
    
    def generate_performance_analysis(self, algorithm: Algorithm) -> PerformanceDoc:
        """
        Generate performance analysis documentation for an algorithm.
        
        Args:
            algorithm: Algorithm object to analyze
            
        Returns:
            PerformanceDoc containing performance analysis
        """
        doc = PerformanceDoc()
        
        # Extract benchmarks from methods
        doc.benchmarks = self._extract_benchmarks(algorithm)
        
        # Generate complexity analysis
        doc.complexity_analysis = self._generate_complexity_analysis(algorithm)
        
        # Generate scalability notes
        doc.scalability_notes = self._generate_scalability_notes(algorithm)
        
        # Generate optimization suggestions
        doc.optimization_suggestions = self._generate_optimization_suggestions(algorithm)
        
        return doc
    
    def _extract_benchmarks(self, algorithm: Algorithm) -> List[Dict[str, Any]]:
        """Extract benchmark information from algorithm methods."""
        benchmarks = []
        
        for method in algorithm.methods:
            method_name = method['name']
            docstring = method.get('docstring', '')
            
            # Look for performance-related information
            performance_info = self._extract_performance_info(method_name, docstring)
            if performance_info:
                benchmarks.append({
                    'method': method_name,
                    'performance_info': performance_info,
                    'complexity': method.get('complexity'),
                    'optimization_notes': self._extract_optimization_notes(docstring)
                })
        
        return benchmarks
    
    def _extract_performance_info(self, method_name: str, docstring: str) -> Dict[str, Any]:
        """Extract performance information from method docstring."""
        performance_info = {}
        
        try:
            # Look for complexity mentions
            complexity_matches = []
            for pattern in self.complexity_patterns:
                matches = re.findall(pattern, docstring, re.IGNORECASE)
                complexity_matches.extend(matches)
            
            if complexity_matches:
                performance_info['complexity'] = complexity_matches
            
            # Look for benchmark mentions
            benchmark_matches = []
            for pattern in self.benchmark_patterns:
                if re.search(pattern, docstring, re.IGNORECASE):
                    # Extract the surrounding context
                    lines = docstring.split('\n')
                    for line in lines:
                        if re.search(pattern, line, re.IGNORECASE):
                            benchmark_matches.append(line.strip())
            
            if benchmark_matches:
                performance_info['benchmarks'] = benchmark_matches
            
            # Extract numerical performance data if present
            numerical_data = re.findall(r'(\d+(?:\.\d+)?)\s*(ms|seconds?|minutes?|hours?|MB|GB|KB)', docstring, re.IGNORECASE)
            if numerical_data:
                performance_info['metrics'] = numerical_data
            
        except Exception as e:
            logger.error(f"Failed to extract benchmarks from {method_name}: {e}")
        
        return performance_info
    
    def _extract_optimization_notes(self, docstring: str) -> List[str]:
        """Extract optimization suggestions from docstring."""
        optimization_keywords = [
            'optimize', 'optimization', 'improve', 'faster', 'efficient',
            'cache', 'memoize', 'parallel', 'vectorize'
        ]
        
        notes = []
        lines = docstring.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in optimization_keywords):
                notes.append(line.strip())
        
        return notes
    
    def _generate_complexity_analysis(self, algorithm: Algorithm) -> str:
        """Generate comprehensive complexity analysis."""
        analysis_parts = []
        
        # Add algorithm-level complexity
        if algorithm.complexity_info:
            analysis_parts.append("## Method Complexity Analysis\n")
            for method_name, complexity in algorithm.complexity_info.items():
                analysis_parts.append(f"**{method_name}**: {complexity}")
        
        # Add overall algorithm complexity if derivable
        complexities = list(algorithm.complexity_info.values())
        if complexities:
            try:
                overall_complexity = self._derive_overall_complexity(complexities)
                analysis_parts.append(f"\n**Overall Algorithm Complexity**: {overall_complexity}")
            except Exception as e:
                logger.error(f"Failed to analyze complexity: {e}")
        
        # Add scalability considerations
        scalability_analysis = self._analyze_scalability(algorithm)
        if scalability_analysis:
            analysis_parts.append(f"\n## Scalability Analysis\n{scalability_analysis}")
        
        return '\n'.join(analysis_parts) if analysis_parts else "Complexity analysis not available."
    
    def _derive_overall_complexity(self, complexities: List[str]) -> str:
        """Derive overall algorithm complexity from method complexities."""
        # Simple heuristic: take the highest complexity
        complexity_order = {
            'O(1)': 1,
            'O(log n)': 2,
            'O(n)': 3,
            'O(n log n)': 4,
            'O(n²)': 5,
            'O(n³)': 6,
            'O(2^n)': 7,
            'O(n!)': 8
        }
        
        max_complexity = 'O(1)'
        max_order = 0
        
        for complexity in complexities:
            # Extract O(...) notation
            o_notation = re.search(r'O\([^)]+\)', complexity)
            if o_notation:
                notation = o_notation.group()
                order = complexity_order.get(notation, 0)
                if order > max_order:
                    max_order = order
                    max_complexity = notation
        
        return max_complexity
    
    def _analyze_scalability(self, algorithm: Algorithm) -> str:
        """Analyze algorithm scalability characteristics."""
        scalability_notes = []
        
        # Check for scalability mentions in docstrings
        all_text = algorithm.docstring + ' '.join(method.get('docstring', '') for method in algorithm.methods)
        
        scalability_keywords = ['scalability', 'scale', 'parallel', 'distributed', 'batch']
        
        for keyword in scalability_keywords:
            if keyword in all_text.lower():
                # Find relevant sentences
                sentences = re.split(r'[.!?]', all_text)
                for sentence in sentences:
                    if keyword in sentence.lower():
                        scalability_notes.append(sentence.strip())
        
        return '\n'.join(scalability_notes) if scalability_notes else ""
    
    def _generate_scalability_notes(self, algorithm: Algorithm) -> str:
        """Generate scalability analysis notes."""
        notes = []
        
        # Analyze based on complexity
        if algorithm.complexity_info:
            for method_name, complexity in algorithm.complexity_info.items():
                if 'O(n²)' in complexity or 'O(2^n)' in complexity:
                    notes.append(f"⚠️ {method_name} has high computational complexity ({complexity}) - consider optimization for large datasets")
                elif 'O(n log n)' in complexity:
                    notes.append(f"✓ {method_name} has good scalability ({complexity}) - suitable for large datasets")
                elif 'O(n)' in complexity:
                    notes.append(f"✓ {method_name} scales linearly ({complexity}) - performance predictable with data size")
                elif 'O(1)' in complexity:
                    notes.append(f"✓ {method_name} has constant time complexity ({complexity}) - excellent scalability")
        
        # Add general scalability recommendations
        if not notes:
            notes.append("Consider profiling with representative datasets to determine scalability characteristics.")
        
        return '\n'.join(notes)
    
    def _generate_optimization_suggestions(self, algorithm: Algorithm) -> List[str]:
        """Generate optimization suggestions based on algorithm analysis."""
        suggestions = []
        
        # Analyze complexity for optimization opportunities
        if algorithm.complexity_info:
            for method_name, complexity in algorithm.complexity_info.items():
                if 'O(n²)' in complexity:
                    suggestions.append(f"Consider algorithmic improvements for {method_name} to reduce quadratic complexity")
                elif 'O(2^n)' in complexity:
                    suggestions.append(f"Exponential complexity in {method_name} - consider memoization or dynamic programming")
        
        # Check for common optimization patterns in docstrings
        all_text = algorithm.docstring + ' '.join(method.get('docstring', '') for method in algorithm.methods)
        
        if 'loop' in all_text.lower():
            suggestions.append("Consider vectorization for loop-heavy operations using NumPy")
        
        if 'cache' in all_text.lower() or 'memoiz' in all_text.lower():
            suggestions.append("Implement caching for expensive computations")
        
        if 'parallel' in all_text.lower():
            suggestions.append("Consider parallel processing for CPU-intensive operations")
        
        # Default suggestions if none found
        if not suggestions:
            suggestions.extend([
                "Profile algorithm performance with representative datasets",
                "Consider caching frequently computed results",
                "Evaluate opportunities for parallel processing"
            ])
        
        return suggestions


class ValidationMethodologyGenerator:
    """Generator for algorithm validation methodology documentation."""
    
    def __init__(self):
        self.validation_patterns = [
            r'test', r'validation', r'verify', r'check', r'assert',
            r'accuracy', r'precision', r'recall', r'metric'
        ]
    
    def generate_validation_spec(self, algorithm: Algorithm) -> ValidationDoc:
        """
        Generate validation methodology specification.
        
        Args:
            algorithm: Algorithm object to generate validation for
            
        Returns:
            ValidationDoc containing validation methodology
        """
        doc = ValidationDoc()
        
        # Extract test methods
        doc.test_methods = self._extract_test_methods(algorithm)
        
        # Generate validation criteria
        doc.validation_criteria = self._generate_validation_criteria(algorithm)
        
        # Extract accuracy metrics
        doc.accuracy_metrics = self._extract_accuracy_metrics(algorithm)
        
        # Identify edge cases
        doc.edge_cases = self._identify_edge_cases(algorithm)
        
        return doc
    
    def _extract_test_methods(self, algorithm: Algorithm) -> List[str]:
        """Extract testing methodologies from algorithm documentation."""
        test_methods = []
        
        # Look for testing mentions in docstrings
        all_text = algorithm.docstring + ' '.join(method.get('docstring', '') for method in algorithm.methods)
        
        # Extract test-related sentences
        sentences = re.split(r'[.!?]', all_text)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(pattern in sentence_lower for pattern in self.validation_patterns):
                test_methods.append(sentence.strip())
        
        # Add standard testing methods if none found
        if not test_methods:
            test_methods.extend([
                "Unit testing for individual method correctness",
                "Integration testing for algorithm workflow",
                "Property-based testing for edge cases"
            ])
        
        return test_methods
    
    def _generate_validation_criteria(self, algorithm: Algorithm) -> List[str]:
        """Generate validation criteria based on algorithm characteristics."""
        criteria = []
        
        # Based on algorithm type and methods
        method_names = [method['name'] for method in algorithm.methods]
        
        if any('predict' in name.lower() for name in method_names):
            criteria.extend([
                "Prediction accuracy on test dataset",
                "Consistency of predictions across runs",
                "Handling of edge cases in input data"
            ])
        
        if any('train' in name.lower() or 'fit' in name.lower() for name in method_names):
            criteria.extend([
                "Convergence of training process",
                "Generalization to unseen data",
                "Stability across different initializations"
            ])
        
        if any('normalize' in name.lower() for name in method_names):
            criteria.extend([
                "Output values within expected range",
                "Preservation of relative ordering",
                "Handling of extreme input values"
            ])
        
        # Default criteria if none specific found
        if not criteria:
            criteria.extend([
                "Correctness of algorithm output",
                "Robustness to input variations",
                "Performance within acceptable bounds"
            ])
        
        return criteria
    
    def _extract_accuracy_metrics(self, algorithm: Algorithm) -> List[str]:
        """Extract accuracy and performance metrics from algorithm documentation."""
        metrics = []
        
        # Look for metric mentions in docstrings
        all_text = algorithm.docstring + ' '.join(method.get('docstring', '') for method in algorithm.methods)
        
        metric_patterns = [
            r'accuracy', r'precision', r'recall', r'f1[- ]score',
            r'mse', r'rmse', r'mae', r'r[²2][- ]score',
            r'auc', r'roc', r'confusion matrix'
        ]
        
        for pattern in metric_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            if matches:
                metrics.extend(matches)
        
        # Add standard metrics based on algorithm type
        if 'classification' in all_text.lower():
            metrics.extend(['Accuracy', 'Precision', 'Recall', 'F1-Score'])
        elif 'regression' in all_text.lower():
            metrics.extend(['MSE', 'RMSE', 'MAE', 'R² Score'])
        elif 'anomaly' in all_text.lower() or 'detection' in all_text.lower():
            metrics.extend(['True Positive Rate', 'False Positive Rate', 'AUC-ROC'])
        
        return list(set(metrics)) if metrics else ['Accuracy', 'Performance', 'Robustness']
    
    def _identify_edge_cases(self, algorithm: Algorithm) -> List[str]:
        """Identify potential edge cases for algorithm testing."""
        edge_cases = []
        
        # Analyze method parameters to identify potential edge cases
        for method in algorithm.methods:
            params = method.get('parameters', [])
            
            for param in params:
                param_name = param['name']
                param_type = param.get('type', '')
                
                if 'array' in param_type.lower() or 'list' in param_type.lower():
                    edge_cases.append(f"Empty {param_name}")
                    edge_cases.append(f"Single-element {param_name}")
                    edge_cases.append(f"Very large {param_name}")
                
                if 'float' in param_type.lower() or 'int' in param_type.lower():
                    edge_cases.append(f"Zero value for {param_name}")
                    edge_cases.append(f"Negative values for {param_name}")
                    edge_cases.append(f"Very large values for {param_name}")
                
                if 'str' in param_type.lower():
                    edge_cases.append(f"Empty string for {param_name}")
                    edge_cases.append(f"Special characters in {param_name}")
        
        # Add algorithm-specific edge cases
        all_text = algorithm.docstring + ' '.join(method.get('docstring', '') for method in algorithm.methods)
        
        if 'division' in all_text.lower() or 'divide' in all_text.lower():
            edge_cases.append("Division by zero scenarios")
        
        if 'matrix' in all_text.lower():
            edge_cases.append("Singular matrices")
            edge_cases.append("Non-square matrices")
        
        if 'convergence' in all_text.lower():
            edge_cases.append("Non-convergent scenarios")
            edge_cases.append("Maximum iteration limits")
        
        return list(set(edge_cases)) if edge_cases else [
            "Boundary value inputs",
            "Invalid input types",
            "Resource exhaustion scenarios"
        ]