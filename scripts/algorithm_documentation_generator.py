#!/usr/bin/env python3
"""
Algorithm Documentation Generator

This module implements IEEE-compliant algorithm documentation generation
with mathematical formulations, complexity analysis, and performance specifications.
"""

import ast
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Algorithm:
    """Represents an algorithm extracted from source code."""
    name: str
    class_name: str
    file_path: Path
    docstring: str = ""
    methods: List[Dict[str, Any]] = field(default_factory=list)
    mathematical_formulations: List[str] = field(default_factory=list)
    complexity_info: Dict[str, str] = field(default_factory=dict)
    performance_metrics: List[str] = field(default_factory=list)
    validation_methods: List[str] = field(default_factory=list)


@dataclass
class MathSpec:
    """Mathematical specification for an algorithm."""
    formulations: List[str] = field(default_factory=list)
    equations: List[str] = field(default_factory=list)
    latex_content: str = ""
    complexity_analysis: str = ""
    variables: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceDoc:
    """Performance analysis documentation."""
    benchmarks: List[Dict[str, Any]] = field(default_factory=list)
    complexity_analysis: str = ""
    scalability_notes: str = ""
    optimization_suggestions: List[str] = field(default_factory=list)


@dataclass
class ValidationDoc:
    """Validation methodology documentation."""
    test_methods: List[str] = field(default_factory=list)
    validation_criteria: List[str] = field(default_factory=list)
    accuracy_metrics: List[str] = field(default_factory=list)
    edge_cases: List[str] = field(default_factory=list)


class AlgorithmGenerator:
    """
    IEEE-compliant algorithm documentation generator.
    
    Generates comprehensive algorithm documentation with mathematical specifications,
    performance analysis, and validation methodologies from service implementations.
    """
    
    def __init__(self, project_root: Path, docs_dir: Path):
        self.project_root = project_root
        self.docs_dir = docs_dir
        self.algorithms_dir = docs_dir / "algorithms"
        self.implementations_dir = self.algorithms_dir / "implementations"
        self.implementations_dir.mkdir(parents=True, exist_ok=True)
        
        # Mathematical notation patterns
        self.math_patterns = {
            'equations': [
                r'[a-zA-Z]\s*=\s*[^=]+',  # Basic equations
                r'f\([^)]+\)\s*=\s*[^=]+',  # Function definitions
                r'∑|∏|∫|∂|∇',  # Mathematical operators
                r'[α-ωΑ-Ω]',  # Greek letters
                r'O\([^)]+\)',  # Big O notation
                r'[≤≥≠≈∈∉⊂⊆∪∩]',  # Mathematical symbols
            ],
            'complexity': [
                r'O\([^)]+\)',  # Big O notation
                r'[Tt]ime [Cc]omplexity',
                r'[Ss]pace [Cc]omplexity',
                r'[Cc]omputational [Cc]omplexity',
            ]
        }
    
    def extract_algorithms(self) -> List[Algorithm]:
        """
        Extract algorithm implementations from service files.
        
        Returns:
            List of Algorithm objects containing extracted information
        """
        algorithms = []
        services_dir = self.project_root / "app" / "services"
        
        if not services_dir.exists():
            logger.warning(f"Services directory not found: {services_dir}")
            return algorithms
        
        # Key algorithm service files to analyze
        algorithm_files = [
            "score_normalizer.py",
            "threshold_manager.py", 
            "embedding_service.py",
            "model_manager.py",
            "interpretability_engine.py",
            "data_pipeline.py",
            "comparison_service.py"
        ]
        
        # Also scan for any other service files that might contain algorithms
        for service_file in services_dir.glob("*.py"):
            if service_file.name not in algorithm_files and service_file.name != "__init__.py":
                algorithm_files.append(service_file.name)
        
        for filename in algorithm_files:
            service_file = services_dir / filename
            if service_file.exists():
                try:
                    file_algorithms = self._extract_algorithms_from_file(service_file)
                    algorithms.extend(file_algorithms)
                except Exception as e:
                    logger.error(f"Failed to extract algorithms from {service_file}: {e}")
        
        logger.info(f"Extracted {len(algorithms)} algorithms from {len(algorithm_files)} service files")
        return algorithms
    
    def _extract_algorithms_from_file(self, file_path: Path) -> List[Algorithm]:
        """Extract algorithms from a single service file."""
        algorithms = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Skip files with null bytes or other encoding issues
            if '\x00' in content:
                logger.warning(f"Skipping file with null bytes: {file_path}")
                return algorithms
            
            tree = ast.parse(content)
            
            # Extract module-level docstring
            module_docstring = ast.get_docstring(tree) or ""
            
            # Find algorithm classes (classes with algorithmic methods)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    algorithm = self._analyze_algorithm_class(node, file_path, content)
                    if algorithm and self._is_algorithm_class(algorithm):
                        algorithms.append(algorithm)
            
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error in {file_path}: {e}")
        except Exception as e:
            logger.error(f"Failed to extract algorithms from {file_path}: {e}")
        
        return algorithms
    
    def _analyze_algorithm_class(self, class_node: ast.ClassDef, file_path: Path, content: str) -> Optional[Algorithm]:
        """Analyze a class to extract algorithm information."""
        class_docstring = ast.get_docstring(class_node) or ""
        
        # Extract methods with their docstrings and complexity info
        methods = []
        mathematical_formulations = []
        complexity_info = {}
        performance_metrics = []
        validation_methods = []
        
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._analyze_method(item, content)
                methods.append(method_info)
                
                # Extract mathematical content from method
                if method_info.get('mathematical_content'):
                    mathematical_formulations.extend(method_info['mathematical_content'])
                
                # Extract complexity information
                if method_info.get('complexity'):
                    complexity_info[method_info['name']] = method_info['complexity']
                
                # Extract performance and validation info
                if method_info.get('performance_info'):
                    performance_metrics.extend(method_info['performance_info'])
                
                if method_info.get('validation_info'):
                    validation_methods.extend(method_info['validation_info'])
        
        # Also extract mathematical content from class docstring
        class_math = self._extract_mathematical_content(class_docstring)
        mathematical_formulations.extend(class_math)
        
        return Algorithm(
            name=class_node.name,
            class_name=class_node.name,
            file_path=file_path,
            docstring=class_docstring,
            methods=methods,
            mathematical_formulations=mathematical_formulations,
            complexity_info=complexity_info,
            performance_metrics=performance_metrics,
            validation_methods=validation_methods
        )
    
    def _analyze_method(self, method_node: ast.FunctionDef, content: str) -> Dict[str, Any]:
        """Analyze a method to extract algorithmic information."""
        method_docstring = ast.get_docstring(method_node) or ""
        
        method_info = {
            'name': method_node.name,
            'docstring': method_docstring,
            'line_number': method_node.lineno,
            'parameters': [],
            'return_annotation': None,
            'mathematical_content': [],
            'complexity': None,
            'performance_info': [],
            'validation_info': []
        }
        
        # Extract parameters with type annotations
        for arg in method_node.args.args:
            param_info = {'name': arg.arg}
            if arg.annotation:
                param_info['type'] = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else str(arg.annotation)
            method_info['parameters'].append(param_info)
        
        # Extract return annotation
        if method_node.returns:
            method_info['return_annotation'] = ast.unparse(method_node.returns) if hasattr(ast, 'unparse') else str(method_node.returns)
        
        # Extract mathematical content from docstring
        if method_docstring:
            method_info['mathematical_content'] = self._extract_mathematical_content(method_docstring)
            method_info['complexity'] = self._extract_complexity_info(method_docstring)
            method_info['performance_info'] = self._extract_performance_info(method_docstring)
            method_info['validation_info'] = self._extract_validation_info(method_docstring)
        
        return method_info
    
    def _extract_mathematical_content(self, text: str) -> List[str]:
        """Extract mathematical formulations from text."""
        mathematical_content = []
        
        # Look for mathematical equations and formulations
        for pattern_type, patterns in self.math_patterns.items():
            if pattern_type == 'equations':
                for pattern in patterns:
                    matches = re.findall(pattern, text, re.MULTILINE)
                    mathematical_content.extend(matches)
        
        # Look for mathematical sections in docstrings
        math_sections = re.findall(
            r'(?:Mathematical|Equation|Formula|Algorithm).*?:\s*\n(.*?)(?=\n\s*[A-Z]|\n\s*$|\Z)',
            text, re.DOTALL | re.IGNORECASE
        )
        mathematical_content.extend(math_sections)
        
        return [content.strip() for content in mathematical_content if content.strip()]
    
    def _extract_complexity_info(self, text: str) -> Optional[str]:
        """Extract computational complexity information from text."""
        for pattern in self.math_patterns['complexity']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Find the surrounding context for the complexity mention
                for match in matches:
                    # Look for the line containing the complexity info
                    lines = text.split('\n')
                    for line in lines:
                        if match.lower() in line.lower():
                            return line.strip()
        return None
    
    def _extract_performance_info(self, text: str) -> List[str]:
        """Extract performance-related information from text."""
        performance_keywords = [
            'performance', 'benchmark', 'speed', 'efficiency', 'optimization',
            'scalability', 'throughput', 'latency', 'memory usage'
        ]
        
        performance_info = []
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in performance_keywords):
                performance_info.append(line.strip())
        
        return performance_info
    
    def _extract_validation_info(self, text: str) -> List[str]:
        """Extract validation methodology information from text."""
        validation_keywords = [
            'validation', 'test', 'verify', 'check', 'accuracy', 'precision',
            'recall', 'metric', 'evaluation', 'assessment'
        ]
        
        validation_info = []
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in validation_keywords):
                validation_info.append(line.strip())
        
        return validation_info
    
    def _is_algorithm_class(self, algorithm: Algorithm) -> bool:
        """Determine if a class represents an algorithm implementation."""
        # Check for algorithmic indicators
        algorithmic_indicators = [
            'algorithm', 'compute', 'calculate', 'process', 'analyze',
            'optimize', 'train', 'predict', 'classify', 'normalize',
            'threshold', 'embedding', 'model', 'service', 'engine'
        ]
        
        name_lower = algorithm.name.lower()
        docstring_lower = algorithm.docstring.lower()
        
        # Check class name and docstring for algorithmic content
        has_algorithmic_name = any(indicator in name_lower for indicator in algorithmic_indicators)
        has_algorithmic_docstring = any(indicator in docstring_lower for indicator in algorithmic_indicators)
        
        # Check for mathematical content
        has_mathematical_content = len(algorithm.mathematical_formulations) > 0
        
        # Check for complexity information
        has_complexity_info = len(algorithm.complexity_info) > 0
        
        # Check for substantial methods (more than just __init__)
        substantial_methods = [m for m in algorithm.methods if not m['name'].startswith('_')]
        has_substantial_methods = len(substantial_methods) > 0
        
        return (has_algorithmic_name or has_algorithmic_docstring or 
                has_mathematical_content or has_complexity_info) and has_substantial_methods
    
    def generate_mathematical_spec(self, algorithm: Algorithm) -> MathSpec:
        """
        Generate mathematical specification for an algorithm.
        
        Args:
            algorithm: Algorithm object to generate specification for
            
        Returns:
            MathSpec object containing mathematical formulations
        """
        spec = MathSpec()
        
        # Extract formulations from algorithm
        spec.formulations = algorithm.mathematical_formulations.copy()
        
        # Generate LaTeX content from formulations
        spec.latex_content = self._generate_latex_content(algorithm)
        
        # Extract and format equations
        spec.equations = self._extract_equations(algorithm)
        
        # Generate complexity analysis
        spec.complexity_analysis = self._generate_complexity_analysis(algorithm)
        
        # Extract mathematical variables and their meanings
        spec.variables = self._extract_mathematical_variables(algorithm)
        
        return spec
    
    def _generate_latex_content(self, algorithm: Algorithm) -> str:
        """Generate LaTeX mathematical notation for the algorithm."""
        latex_content = []
        
        # Add algorithm overview
        latex_content.append(f"\\section{{{algorithm.name} Algorithm}}")
        
        if algorithm.docstring:
            # Extract mathematical sections from docstring
            math_sections = re.findall(
                r'(?:Mathematical|Equation|Formula).*?:\s*\n(.*?)(?=\n\s*[A-Z]|\n\s*$|\Z)',
                algorithm.docstring, re.DOTALL | re.IGNORECASE
            )
            
            for section in math_sections:
                # Convert common mathematical notation to LaTeX
                latex_section = self._convert_to_latex(section)
                latex_content.append(f"\\begin{{equation}}\n{latex_section}\n\\end{{equation}}")
        
        # Add method-specific mathematical content
        for method in algorithm.methods:
            if method.get('mathematical_content'):
                latex_content.append(f"\\subsection{{{method['name']} Method}}")
                for math_content in method['mathematical_content']:
                    latex_math = self._convert_to_latex(math_content)
                    latex_content.append(f"\\begin{{align}}\n{latex_math}\n\\end{{align}}")
        
        return '\n\n'.join(latex_content)
    
    def _convert_to_latex(self, text: str) -> str:
        """Convert mathematical notation to LaTeX format."""
        # Basic conversions for common mathematical symbols
        conversions = {
            'Σ': r'\sum',
            '∑': r'\sum',
            '∏': r'\prod',
            '∫': r'\int',
            '∂': r'\partial',
            '∇': r'\nabla',
            '∞': r'\infty',
            '≤': r'\leq',
            '≥': r'\geq',
            '≠': r'\neq',
            '≈': r'\approx',
            '∈': r'\in',
            '∉': r'\notin',
            '⊂': r'\subset',
            '⊆': r'\subseteq',
            '∪': r'\cup',
            '∩': r'\cap',
            'α': r'\alpha',
            'β': r'\beta',
            'γ': r'\gamma',
            'δ': r'\delta',
            'ε': r'\epsilon',
            'ζ': r'\zeta',
            'η': r'\eta',
            'θ': r'\theta',
            'λ': r'\lambda',
            'μ': r'\mu',
            'ν': r'\nu',
            'ξ': r'\xi',
            'π': r'\pi',
            'ρ': r'\rho',
            'σ': r'\sigma',
            'τ': r'\tau',
            'φ': r'\phi',
            'χ': r'\chi',
            'ψ': r'\psi',
            'ω': r'\omega',
            'ℝ': r'\mathbb{R}',
            'ℕ': r'\mathbb{N}',
            'ℤ': r'\mathbb{Z}',
            'ℚ': r'\mathbb{Q}',
            'ℂ': r'\mathbb{C}',
        }
        
        latex_text = text
        for symbol, latex_symbol in conversions.items():
            latex_text = latex_text.replace(symbol, latex_symbol)
        
        # Convert subscripts and superscripts
        latex_text = re.sub(r'([a-zA-Z])_([a-zA-Z0-9]+)', r'\1_{\2}', latex_text)
        latex_text = re.sub(r'([a-zA-Z])\^([a-zA-Z0-9]+)', r'\1^{\2}', latex_text)
        
        # Convert fractions
        latex_text = re.sub(r'(\d+)/(\d+)', r'\\frac{\1}{\2}', latex_text)
        
        return latex_text
    
    def _extract_equations(self, algorithm: Algorithm) -> List[str]:
        """Extract and format mathematical equations."""
        equations = []
        
        # Extract from mathematical formulations
        for formulation in algorithm.mathematical_formulations:
            # Look for equation patterns
            equation_matches = re.findall(r'[a-zA-Z_]\w*\s*=\s*[^=\n]+', formulation)
            equations.extend(equation_matches)
        
        # Extract from method docstrings
        for method in algorithm.methods:
            if method.get('mathematical_content'):
                for content in method['mathematical_content']:
                    equation_matches = re.findall(r'[a-zA-Z_]\w*\s*=\s*[^=\n]+', content)
                    equations.extend(equation_matches)
        
        return [eq.strip() for eq in equations if eq.strip()]
    
    def _generate_complexity_analysis(self, algorithm: Algorithm) -> str:
        """Generate computational complexity analysis."""
        complexity_parts = []
        
        # Collect complexity information from all methods
        for method_name, complexity in algorithm.complexity_info.items():
            complexity_parts.append(f"**{method_name}**: {complexity}")
        
        # Add general complexity analysis if available
        if algorithm.docstring:
            complexity_sections = re.findall(
                r'(?:Complexity|Performance).*?:\s*\n(.*?)(?=\n\s*[A-Z]|\n\s*$|\Z)',
                algorithm.docstring, re.DOTALL | re.IGNORECASE
            )
            complexity_parts.extend(complexity_sections)
        
        return '\n'.join(complexity_parts) if complexity_parts else "Complexity analysis not available."
    
    def _extract_mathematical_variables(self, algorithm: Algorithm) -> Dict[str, str]:
        """Extract mathematical variables and their meanings."""
        variables = {}
        
        # Look for variable definitions in docstrings
        all_text = algorithm.docstring + ' '.join(method.get('docstring', '') for method in algorithm.methods)
        
        # Pattern to match variable definitions like "where x is the input"
        variable_patterns = [
            r'(?:where|Where)\s+([a-zA-Z_]\w*)\s+(?:is|are|represents?)\s+([^.\n]+)',
            r'([a-zA-Z_]\w*)\s*:\s*([^.\n]+)',
            r'([a-zA-Z_]\w*)\s+(?:is|are|represents?)\s+([^.\n]+)'
        ]
        
        for pattern in variable_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            for var_name, description in matches:
                if len(var_name) <= 3:  # Likely a mathematical variable
                    variables[var_name.strip()] = description.strip()
        
        return variables
    
    def generate_performance_analysis(self, algorithm: Algorithm) -> PerformanceDoc:
        """
        Generate performance analysis for an algorithm.
        
        Args:
            algorithm: Algorithm object to analyze
            
        Returns:
            PerformanceDoc containing performance analysis
        """
        from .performance_analysis_generator import PerformanceAnalysisGenerator
        
        generator = PerformanceAnalysisGenerator()
        return generator.generate_performance_analysis(algorithm)
    
    def generate_validation_spec(self, algorithm: Algorithm) -> ValidationDoc:
        """
        Generate validation methodology specification for an algorithm.
        
        Args:
            algorithm: Algorithm object to generate validation for
            
        Returns:
            ValidationDoc containing validation methodology
        """
        from .performance_analysis_generator import ValidationMethodologyGenerator
        
        generator = ValidationMethodologyGenerator()
        return generator.generate_validation_spec(algorithm)
    
    def generate_algorithm_documentation(self, algorithm: Algorithm) -> str:
        """
        Generate comprehensive IEEE-compliant documentation for an algorithm.
        
        Args:
            algorithm: Algorithm object to document
            
        Returns:
            Complete markdown documentation string
        """
        # Generate mathematical specification
        math_spec = self.generate_mathematical_spec(algorithm)
        
        # Generate performance analysis
        performance_doc = self.generate_performance_analysis(algorithm)
        
        # Generate validation specification
        validation_doc = self.generate_validation_spec(algorithm)
        
        # Build comprehensive documentation
        doc_content = self._build_documentation_content(
            algorithm, math_spec, performance_doc, validation_doc
        )
        
        return doc_content
    
    def _build_documentation_content(self, algorithm: Algorithm, math_spec: MathSpec, 
                                   performance_doc: PerformanceDoc, validation_doc: ValidationDoc) -> str:
        """Build the complete documentation content."""
        content = []
        
        # Header
        content.append(f"# {algorithm.name} Algorithm Implementation")
        content.append("")
        content.append(f"**Source File**: `{algorithm.file_path.relative_to(self.project_root)}`")
        content.append(f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("")
        
        # Overview
        content.append("## Overview")
        content.append("")
        if algorithm.docstring:
            content.append(algorithm.docstring)
        else:
            content.append(f"Implementation of the {algorithm.name} algorithm.")
        content.append("")
        
        # Mathematical Formulation
        if math_spec.formulations or math_spec.equations:
            content.append("## Mathematical Formulation")
            content.append("")
            content.append("*This section provides the mathematical foundation and formal specification of the algorithm.*")
            content.append("")
            
            if math_spec.equations:
                content.append("### Core Equations")
                content.append("")
                for equation in math_spec.equations:
                    content.append(f"```")
                    content.append(equation)
                    content.append("```")
                content.append("")
            
            if math_spec.formulations:
                content.append("### Mathematical Formulations")
                content.append("")
                for formulation in math_spec.formulations:
                    content.append(formulation)
                    content.append("")
            
            if math_spec.variables:
                content.append("### Variable Definitions")
                content.append("")
                for var, description in math_spec.variables.items():
                    content.append(f"- **{var}**: {description}")
                content.append("")
        
        # Computational Complexity
        content.append("## Computational Complexity Analysis")
        content.append("")
        content.append("*This section analyzes the time and space complexity characteristics of the algorithm.*")
        content.append("")
        
        if math_spec.complexity_analysis:
            content.append(math_spec.complexity_analysis)
        else:
            content.append("Complexity analysis not available.")
        content.append("")
        
        # Performance Analysis
        content.append("## Performance Analysis")
        content.append("")
        content.append("*This section provides performance benchmarks and scalability characteristics.*")
        content.append("")
        
        if performance_doc.benchmarks:
            content.append("### Performance Benchmarks")
            content.append("")
            for benchmark in performance_doc.benchmarks:
                method_name = benchmark.get('method', 'Unknown')
                content.append(f"#### {method_name}")
                
                perf_info = benchmark.get('performance_info', {})
                if perf_info.get('complexity'):
                    content.append(f"**Complexity**: {', '.join(perf_info['complexity'])}")
                
                if perf_info.get('benchmarks'):
                    content.append("**Performance Notes**:")
                    for note in perf_info['benchmarks']:
                        content.append(f"- {note}")
                
                if perf_info.get('metrics'):
                    content.append("**Metrics**:")
                    for value, unit in perf_info['metrics']:
                        content.append(f"- {value} {unit}")
                
                content.append("")
        
        if performance_doc.scalability_notes:
            content.append("### Scalability Analysis")
            content.append("")
            content.append(performance_doc.scalability_notes)
            content.append("")
        
        if performance_doc.optimization_suggestions:
            content.append("### Optimization Recommendations")
            content.append("")
            for suggestion in performance_doc.optimization_suggestions:
                content.append(f"- {suggestion}")
            content.append("")
        
        # Validation Methodology
        content.append("## Validation Methodology")
        content.append("")
        content.append("*This section describes the testing and validation approach for the algorithm.*")
        content.append("")
        
        if validation_doc.test_methods:
            content.append("### Testing Methods")
            content.append("")
            for method in validation_doc.test_methods:
                content.append(f"- {method}")
            content.append("")
        
        if validation_doc.validation_criteria:
            content.append("### Validation Criteria")
            content.append("")
            for criterion in validation_doc.validation_criteria:
                content.append(f"- {criterion}")
            content.append("")
        
        if validation_doc.accuracy_metrics:
            content.append("### Accuracy Metrics")
            content.append("")
            for metric in validation_doc.accuracy_metrics:
                content.append(f"- {metric}")
            content.append("")
        
        if validation_doc.edge_cases:
            content.append("### Edge Cases")
            content.append("")
            content.append("The following edge cases should be tested:")
            content.append("")
            for edge_case in validation_doc.edge_cases:
                content.append(f"- {edge_case}")
            content.append("")
        
        # Implementation Details
        content.append("## Implementation Details")
        content.append("")
        
        if algorithm.methods:
            content.append("### Methods")
            content.append("")
            
            for method in algorithm.methods:
                if not method['name'].startswith('_'):  # Skip private methods
                    content.append(f"#### `{method['name']}`")
                    content.append("")
                    
                    if method.get('docstring'):
                        content.append(method['docstring'])
                        content.append("")
                    
                    # Parameters
                    if method.get('parameters'):
                        content.append("**Parameters:**")
                        for param in method['parameters']:
                            param_type = param.get('type', 'Any')
                            content.append(f"- `{param['name']}` ({param_type})")
                        content.append("")
                    
                    # Return type
                    if method.get('return_annotation'):
                        content.append(f"**Returns:** {method['return_annotation']}")
                        content.append("")
        
        # LaTeX Mathematical Notation
        if math_spec.latex_content:
            content.append("## LaTeX Mathematical Notation")
            content.append("")
            content.append("*For formal mathematical documentation and publication:*")
            content.append("")
            content.append("```latex")
            content.append(math_spec.latex_content)
            content.append("```")
            content.append("")
        
        # References and Standards
        content.append("## References and Standards")
        content.append("")
        content.append("- **IEEE Standard 830-1998**: Software Requirements Specifications")
        content.append("- **Algorithm Documentation**: Following IEEE guidelines for algorithm specification")
        content.append("- **Mathematical Notation**: Standard mathematical notation and LaTeX formatting")
        content.append("")
        
        # Footer
        content.append("---")
        content.append("")
        content.append("*This documentation was automatically generated from source code analysis.*")
        content.append(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        return '\n'.join(content)
    
    def generate_comprehensive_algorithm_docs(self) -> Dict[str, Any]:
        """
        Generate comprehensive algorithm documentation for all algorithms.
        
        Returns:
            Dictionary containing generation results and statistics
        """
        logger.info("🧮 Generating comprehensive algorithm documentation...")
        
        results = {
            'algorithms_documented': 0,
            'mathematical_formulations': 0,
            'complexity_analyses': 0,
            'files_generated': 0,
            'errors': []
        }
        
        try:
            # Extract all algorithms
            algorithms = self.extract_algorithms()
            
            if not algorithms:
                logger.warning("No algorithms found to document")
                return results
            
            # Generate documentation for each algorithm
            for algorithm in algorithms:
                try:
                    # Generate comprehensive documentation
                    doc_content = self.generate_algorithm_documentation(algorithm)
                    
                    # Save to file
                    output_file = self.implementations_dir / f"{algorithm.file_path.stem}.md"
                    output_file.write_text(doc_content, encoding='utf-8')
                    
                    # Update statistics
                    results['algorithms_documented'] += 1
                    results['files_generated'] += 1
                    
                    if algorithm.mathematical_formulations:
                        results['mathematical_formulations'] += len(algorithm.mathematical_formulations)
                    
                    if algorithm.complexity_info:
                        results['complexity_analyses'] += len(algorithm.complexity_info)
                    
                    logger.debug(f"Generated documentation for {algorithm.name}")
                    
                except Exception as e:
                    error_msg = f"Failed to generate documentation for {algorithm.name}: {str(e)}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
            
            logger.info(f"  ✅ Algorithm documentation generated:")
            logger.info(f"     - Algorithms documented: {results['algorithms_documented']}")
            logger.info(f"     - Mathematical formulations: {results['mathematical_formulations']}")
            logger.info(f"     - Complexity analyses: {results['complexity_analyses']}")
            logger.info(f"     - Files generated: {results['files_generated']}")
            
            if results['errors']:
                logger.warning(f"     - Errors encountered: {len(results['errors'])}")
            
        except Exception as e:
            error_msg = f"Algorithm documentation generation failed: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        return results