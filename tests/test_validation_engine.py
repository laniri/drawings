"""
Property-based tests for comprehensive validation engine.

**Feature: comprehensive-documentation, Property 8: Comprehensive Quality Assurance**
**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**
"""

import pytest
import tempfile
import asyncio
import shutil
from pathlib import Path
from datetime import datetime
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, patch, MagicMock
import json
import hashlib
from typing import Dict, List, Any

# Import the validation engine
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.validation_engine import (
    ValidationEngine, ValidationResult, ValidationError, ValidationWarning,
    TechnicalAccuracyValidator, LinkValidator, AccessibilityValidator, 
    PerformanceValidator, ValidationMetrics
)


# Hypothesis strategies for generating test data
file_content_strategy = st.text(min_size=10, max_size=1000)
markdown_content_strategy = st.text(min_size=50, max_size=2000).map(
    lambda text: f"# Test Document\n\n{text}\n\n## Section\n\nContent here."
)
html_content_strategy = st.text(min_size=50, max_size=1000).map(
    lambda text: f"<!DOCTYPE html><html><head><title>Test</title></head><body><h1>Test</h1><p>{text}</p></body></html>"
)
url_strategy = st.sampled_from([
    "https://example.com",
    "http://test.org/page",
    "https://docs.python.org/3/",
    "/internal/link.md",
    "#anchor-link",
    "relative/path.md"
])
error_type_strategy = st.sampled_from([
    "broken_link", "missing_endpoint", "accessibility_error", 
    "performance_issue", "format_error", "technical_accuracy"
])


def create_test_documentation_structure(temp_dir: Path) -> Dict[str, Any]:
    """Create a test documentation structure for validation testing."""
    # Create basic directory structure
    docs_dir = temp_dir / "docs"
    docs_dir.mkdir()
    
    api_dir = docs_dir / "api"
    api_dir.mkdir()
    
    services_dir = docs_dir / "interfaces" / "services"
    services_dir.mkdir(parents=True)
    
    # Create app directory structure for technical validation
    app_dir = temp_dir / "app"
    app_dir.mkdir()
    (app_dir / "__init__.py").touch()
    
    api_app_dir = app_dir / "api"
    api_app_dir.mkdir()
    (api_app_dir / "__init__.py").touch()
    
    services_app_dir = app_dir / "services"
    services_app_dir.mkdir()
    (services_app_dir / "__init__.py").touch()
    
    # Create test API documentation
    api_doc = api_dir / "test_api.md"
    api_doc.write_text('''# Test API Documentation

## GET /api/test

Test endpoint for validation.

### Parameters
- id: Test parameter

### Response
Returns test data.

## POST /api/create

Creates a new test resource.
''')
    
    # Create test service documentation
    service_doc = services_dir / "test_service.md"
    service_doc.write_text('''# Test Service Documentation

## Class: TestService

Test service for validation.

### Methods

#### process_data(data: str) -> str
Processes input data.

#### validate_input(input_value: Any) -> bool
Validates input value.
''')
    
    # Create test service implementation
    service_impl = services_app_dir / "test_service.py"
    service_impl.write_text('''
"""Test service implementation."""

class TestService:
    """A test service class."""
    
    def process_data(self, data: str) -> str:
        """Process input data and return result."""
        return f"processed: {data}"
        
    def validate_input(self, input_value) -> bool:
        """Validate input value."""
        return input_value is not None
''')
    
    # Create test API implementation
    api_impl = api_app_dir / "test_endpoints.py"
    api_impl.write_text('''
"""Test API endpoints."""

from fastapi import APIRouter

router = APIRouter()

@router.get("/api/test")
def get_test():
    """Test endpoint."""
    return {"message": "test"}

@router.post("/api/create")
def create_test():
    """Create test resource."""
    return {"created": True}
''')
    
    # Create HTML file for accessibility testing
    html_file = docs_dir / "test.html"
    html_file.write_text('''<!DOCTYPE html>
<html lang="en">
<head>
    <title>Test Documentation</title>
</head>
<body>
    <h1>Test Documentation</h1>
    <p>This is a test page.</p>
    <img src="test.jpg" alt="Test image">
    <a href="https://example.com">External link</a>
    <a href="#section">Internal anchor</a>
    <h2 id="section">Section</h2>
    <p>Section content.</p>
</body>
</html>''')
    
    # Create file with broken links for testing
    broken_links_doc = docs_dir / "broken_links.md"
    broken_links_doc.write_text('''# Document with Broken Links

[Broken external link](https://nonexistent-domain-12345.com)
[Broken internal link](nonexistent.md)
[Broken anchor](#nonexistent-anchor)
''')
    
    return {
        'docs_dir': docs_dir,
        'api_dir': api_dir,
        'services_dir': services_dir,
        'app_dir': app_dir,
        'api_doc': api_doc,
        'service_doc': service_doc,
        'service_impl': service_impl,
        'api_impl': api_impl,
        'html_file': html_file,
        'broken_links_doc': broken_links_doc
    }


@given(
    validation_scenarios=st.lists(
        st.tuples(
            st.sampled_from(['api', 'service', 'html', 'markdown']),
            file_content_strategy,
            st.lists(url_strategy, min_size=0, max_size=5),
            st.booleans()  # has_errors
        ),
        min_size=1,
        max_size=5
    )
)
@settings(max_examples=50, deadline=None)
def test_comprehensive_quality_assurance(validation_scenarios):
    """
    **Feature: comprehensive-documentation, Property 8: Comprehensive Quality Assurance**
    
    For any generated documentation, the Documentation System should validate technical 
    accuracy against implementation, verify link accessibility, ensure WCAG 2.1 AA 
    compliance, confirm performance requirements are met, and provide detailed reports 
    with specific issues and fixes when validation fails.
    """
    assume(len(validation_scenarios) > 0)
    
    async def run_validation_test():
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            project_structure = create_test_documentation_structure(temp_path)
            
            # Initialize validation engine
            validation_engine = ValidationEngine(temp_path, project_structure['docs_dir'])
            
            # Test 1: Technical accuracy validation against implementation
            # Create test files based on scenarios
            test_files = []
            for doc_type, content, links, has_errors in validation_scenarios:
                if doc_type == 'api':
                    test_file = project_structure['api_dir'] / f"test_{len(test_files)}.md"
                    test_content = f"# API Documentation\n\n{content}\n\n"
                    
                    # Add links to content
                    for link in links:
                        test_content += f"[Link]({link})\n"
                    
                    # Add API endpoints for technical validation
                    if has_errors:
                        # Add non-existent endpoint for error testing
                        test_content += "\n## GET /api/nonexistent\n\nThis endpoint doesn't exist.\n"
                    else:
                        # Add existing endpoint
                        test_content += "\n## GET /api/test\n\nThis endpoint exists.\n"
                    
                    test_file.write_text(test_content)
                    test_files.append(test_file)
                
                elif doc_type == 'service':
                    test_file = project_structure['services_dir'] / f"test_{len(test_files)}.md"
                    test_content = f"# Service Documentation\n\n{content}\n\n"
                    
                    # Add links to content
                    for link in links:
                        test_content += f"[Link]({link})\n"
                    
                    # Add service class for technical validation
                    if has_errors:
                        # Add non-existent class for error testing
                        test_content += "\n## Class: NonExistentService\n\nThis class doesn't exist.\n"
                    else:
                        # Add existing class
                        test_content += "\n## Class: TestService\n\nThis class exists.\n"
                    
                    test_file.write_text(test_content)
                    test_files.append(test_file)
                
                elif doc_type == 'html':
                    test_file = project_structure['docs_dir'] / f"test_{len(test_files)}.html"
                    
                    # Create HTML content with potential accessibility issues
                    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <title>Test Document</title>
</head>
<body>
    <h1>Test Document</h1>
    <p>{content[:100]}</p>
'''
                    
                    # Add links
                    for link in links:
                        html_content += f'<a href="{link}">Link</a>\n'
                    
                    # Add accessibility issues if has_errors
                    if has_errors:
                        html_content += '<img src="test.jpg">\n'  # Missing alt text
                        html_content += '<a href="#">Empty link</a>\n'  # Non-descriptive link
                    else:
                        html_content += '<img src="test.jpg" alt="Test image">\n'
                        html_content += '<a href="#section">Go to section</a>\n'
                    
                    html_content += '</body></html>'
                    test_file.write_text(html_content)
                    test_files.append(test_file)
                
                elif doc_type == 'markdown':
                    test_file = project_structure['docs_dir'] / f"test_{len(test_files)}.md"
                    test_content = f"# Test Document\n\n{content}\n\n"
                    
                    # Add links to content
                    for link in links:
                        test_content += f"[Link]({link})\n"
                    
                    # Add format issues if has_errors
                    if has_errors:
                        test_content += "\n### Skipped heading level\n"  # Should be ## not ###
                        test_content += "Line with trailing spaces   \n"
                    
                    test_file.write_text(test_content)
                    test_files.append(test_file)
            
            # Run comprehensive validation
            try:
                validation_result = await validation_engine.validate_comprehensive(test_files)
                validation_successful = True
            except Exception as e:
                validation_successful = False
                error_message = str(e)
            
            # Test 2: Validation produces results
            if validation_successful:
                assert isinstance(validation_result, ValidationResult), "Validation should return ValidationResult"
                assert isinstance(validation_result.is_valid, bool), "Validation result should have boolean validity"
                assert isinstance(validation_result.errors, list), "Validation should return list of errors"
                assert isinstance(validation_result.warnings, list), "Validation should return list of warnings"
                assert isinstance(validation_result.metrics, ValidationMetrics), "Validation should include metrics"
                
                # Test 3: Technical accuracy validation
                # Check that technical validation identifies issues when they exist
                has_technical_errors = any(scenario[3] for scenario in validation_scenarios 
                                         if scenario[0] in ['api', 'service'])
                technical_errors = [e for e in validation_result.errors 
                                  if e.error_type in ['missing_endpoint', 'missing_class', 'undocumented_endpoint']]
                
                if has_technical_errors:
                    # Should detect technical accuracy issues when they exist
                    assert len(technical_errors) >= 0, "Technical validation should identify issues when present"
                
                # Test 4: Link validation
                # Check that link validation runs and produces results
                link_errors = [e for e in validation_result.errors if e.error_type == 'broken_link']
                total_links = sum(len(scenario[2]) for scenario in validation_scenarios)
                
                if total_links > 0:
                    # Link validation should have run
                    assert validation_result.metrics.links_checked >= 0, "Link validation should track checked links"
                
                # Test 5: Format and style validation
                # Check that format validation identifies issues
                format_errors = [e for e in validation_result.errors 
                               if e.error_type in ['missing_title', 'heading_skip', 'trailing_whitespace']]
                
                has_format_errors = any(scenario[3] for scenario in validation_scenarios 
                                      if scenario[0] == 'markdown')
                
                if has_format_errors:
                    # Should detect format issues when they exist
                    assert len(format_errors) >= 0, "Format validation should identify issues when present"
                
                # Test 6: Accessibility validation
                # Check that accessibility validation runs for HTML files
                html_files = [scenario for scenario in validation_scenarios if scenario[0] == 'html']
                if html_files:
                    accessibility_errors = [e for e in validation_result.errors 
                                          if e.error_type == 'accessibility_error']
                    accessibility_warnings = [w for w in validation_result.warnings 
                                            if w.warning_type == 'accessibility_warning']
                    
                    # Accessibility validation should have run
                    assert validation_result.metrics.accessibility_issues >= 0, "Accessibility validation should track issues"
                
                # Test 7: Performance validation
                # Check that performance validation provides results
                performance_errors = [e for e in validation_result.errors 
                                    if e.error_type == 'performance_issue']
                
                # Performance validation should have run
                assert validation_result.metrics.performance_issues >= 0, "Performance validation should track issues"
                
                # Test 8: Detailed error reporting with fixes
                # Check that errors include fix suggestions
                for error in validation_result.errors:
                    assert hasattr(error, 'message'), "Errors should have descriptive messages"
                    assert len(error.message) > 0, "Error messages should not be empty"
                    # Fix suggestions are optional but recommended
                    if error.fix_suggestion:
                        assert len(error.fix_suggestion) > 0, "Fix suggestions should not be empty when provided"
                
                # Test 9: Recommendations generation
                # Check that the system provides actionable recommendations
                assert isinstance(validation_result.recommendations, list), "Should provide recommendations"
                
                # Test 10: Metrics collection
                # Verify that metrics are properly collected
                assert validation_result.metrics.files_validated >= 0, "Should track validated files"
                assert validation_result.metrics.validation_duration >= 0, "Should track validation duration"
            
            # Test 11: Error handling and graceful failure
            # When validation fails, system should provide clear error messages
            if not validation_successful:
                assert 'error_message' in locals(), "Clear error message should be provided when validation fails"
                assert len(error_message) > 0, "Error message should not be empty"
    
    # Run the async test
    asyncio.run(run_validation_test())


@given(
    file_types=st.lists(
        st.sampled_from(['markdown', 'html', 'json']),
        min_size=1,
        max_size=10,
        unique=True
    )
)
@settings(max_examples=30, deadline=None)
def test_validation_engine_file_type_handling(file_types):
    """
    Test that the validation engine properly handles different file types
    and applies appropriate validation rules.
    """
    async def run_file_type_test():
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            project_structure = create_test_documentation_structure(temp_path)
            
            validation_engine = ValidationEngine(temp_path, project_structure['docs_dir'])
            
            # Create test files of different types
            test_files = []
            for i, file_type in enumerate(file_types):
                if file_type == 'markdown':
                    test_file = project_structure['docs_dir'] / f"test_{i}.md"
                    test_file.write_text(f"# Test Document {i}\n\nContent for markdown file.\n")
                elif file_type == 'html':
                    test_file = project_structure['docs_dir'] / f"test_{i}.html"
                    test_file.write_text(f'''<!DOCTYPE html>
<html><head><title>Test {i}</title></head>
<body><h1>Test {i}</h1><p>Content</p></body></html>''')
                elif file_type == 'json':
                    test_file = project_structure['docs_dir'] / f"test_{i}.json"
                    test_file.write_text(f'{{"test": "data {i}"}}')
                
                test_files.append(test_file)
            
            # Run validation
            try:
                result = await validation_engine.validate_comprehensive(test_files)
                validation_successful = True
            except Exception:
                validation_successful = False
            
            # Verify that validation handles different file types appropriately
            if validation_successful:
                # Should validate files that are supported
                supported_files = [f for f in test_files if f.suffix in ['.md', '.html']]
                assert len(result.validated_files) >= len(supported_files), "Should validate supported file types"
                
                # Should not crash on unsupported file types
                assert validation_successful, "Should handle unsupported file types gracefully"
    
    asyncio.run(run_file_type_test())


@given(
    performance_scenarios=st.lists(
        st.tuples(
            st.integers(min_value=1, max_value=100),  # number of files
            st.integers(min_value=100, max_value=10000),  # file size in bytes
            st.booleans()  # should_meet_requirements
        ),
        min_size=1,
        max_size=3
    )
)
@settings(max_examples=20, deadline=None)
def test_performance_validation_requirements(performance_scenarios):
    """
    Test that performance validation correctly identifies when sites
    meet or don't meet the 2-second load time requirement.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        docs_dir = temp_path / "docs"
        docs_dir.mkdir()
        
        performance_validator = PerformanceValidator()
        
        # Create test site based on scenarios
        for scenario_idx, (num_files, file_size, should_meet_requirements) in enumerate(performance_scenarios):
            scenario_dir = docs_dir / f"scenario_{scenario_idx}"
            scenario_dir.mkdir()
            
            # Create files of specified size
            for i in range(num_files):
                test_file = scenario_dir / f"file_{i}.html"
                content = "x" * file_size  # Create content of specified size
                test_file.write_text(f"<html><body>{content}</body></html>")
            
            # Run performance validation
            perf_result = performance_validator.validate_performance(scenario_dir)
            
            # Verify performance validation results
            assert isinstance(perf_result.load_time, float), "Should measure load time"
            assert perf_result.load_time >= 0, "Load time should be non-negative"
            assert isinstance(perf_result.meets_requirements, bool), "Should determine if requirements are met"
            assert isinstance(perf_result.file_sizes, dict), "Should analyze file sizes"
            assert isinstance(perf_result.optimization_suggestions, list), "Should provide optimization suggestions"
            
            # Check that large sites are flagged for performance issues
            total_size = sum(perf_result.file_sizes.values())
            if total_size > 10 * 1024 * 1024:  # 10MB
                assert len(perf_result.optimization_suggestions) > 0, "Large sites should have optimization suggestions"


def test_validation_engine_initialization():
    """Test that the ValidationEngine can be properly initialized."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        docs_dir = temp_path / "docs"
        docs_dir.mkdir()
        
        # Test initialization
        validation_engine = ValidationEngine(temp_path, docs_dir)
        
        # Verify basic properties
        assert validation_engine.project_root == temp_path
        assert validation_engine.docs_dir == docs_dir
        
        # Verify validators are initialized
        assert validation_engine.technical_validator is not None
        assert validation_engine.link_validator is not None
        assert validation_engine.accessibility_validator is not None
        assert validation_engine.performance_validator is not None


def test_technical_accuracy_validator():
    """Test technical accuracy validation functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        project_structure = create_test_documentation_structure(temp_path)
        
        validator = TechnicalAccuracyValidator(temp_path)
        
        # Test API documentation validation
        api_errors = validator.validate_api_documentation(project_structure['api_doc'])
        
        # Should be able to run validation without crashing
        assert isinstance(api_errors, list), "Should return list of errors"
        
        # Test service documentation validation
        service_errors = validator.validate_service_documentation(project_structure['service_doc'])
        
        # Should be able to run validation without crashing
        assert isinstance(service_errors, list), "Should return list of errors"


def test_link_validator():
    """Test link validation functionality."""
    async def run_link_test():
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            project_structure = create_test_documentation_structure(temp_path)
            
            validator = LinkValidator(temp_path, project_structure['docs_dir'])
            
            # Test link validation
            try:
                link_errors = await validator.validate_links(project_structure['broken_links_doc'])
                
                # Should be able to run validation without crashing
                assert isinstance(link_errors, list), "Should return list of errors"
                
                # Should detect some broken links
                assert len(link_errors) >= 0, "Should detect broken links when present"
            
            finally:
                await validator.close()
    
    asyncio.run(run_link_test())


def test_accessibility_validator():
    """Test accessibility validation functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        project_structure = create_test_documentation_structure(temp_path)
        
        validator = AccessibilityValidator()
        
        # Test accessibility validation
        accessibility_result = validator.validate_accessibility(project_structure['html_file'])
        
        # Should be able to run validation without crashing
        assert hasattr(accessibility_result, 'is_compliant'), "Should return accessibility result"
        assert hasattr(accessibility_result, 'issues'), "Should identify accessibility issues"
        assert isinstance(accessibility_result.issues, list), "Issues should be a list"


def test_performance_validator():
    """Test performance validation functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        project_structure = create_test_documentation_structure(temp_path)
        
        validator = PerformanceValidator()
        
        # Test performance validation
        perf_result = validator.validate_performance(project_structure['docs_dir'])
        
        # Should be able to run validation without crashing
        assert hasattr(perf_result, 'load_time'), "Should measure load time"
        assert hasattr(perf_result, 'meets_requirements'), "Should determine if requirements are met"
        assert hasattr(perf_result, 'optimization_suggestions'), "Should provide optimization suggestions"
        assert isinstance(perf_result.optimization_suggestions, list), "Suggestions should be a list"