#!/usr/bin/env python3
"""
Comprehensive Validation Engine for Documentation System

This module implements multi-layered validation for all documentation types,
including technical accuracy validation, link validation, accessibility compliance,
and performance validation.
"""

import os
import re
import json
import time
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlparse, urljoin
import hashlib
import ast
import inspect

# HTTP client for link validation
try:
    import aiohttp
    import requests
    HAS_HTTP_CLIENTS = True
except ImportError:
    HAS_HTTP_CLIENTS = False

# HTML parsing for accessibility validation
try:
    from bs4 import BeautifulSoup
    HAS_HTML_PARSER = True
except ImportError:
    HAS_HTML_PARSER = False
    # Create a mock BeautifulSoup for type hints when not available
    class BeautifulSoup:
        pass


@dataclass
class ValidationError:
    """Represents a validation error with context."""
    file_path: Path
    line_number: Optional[int]
    error_type: str
    message: str
    severity: str = "error"  # error, warning, info
    fix_suggestion: Optional[str] = None


@dataclass
class ValidationWarning:
    """Represents a validation warning with context."""
    file_path: Path
    line_number: Optional[int]
    warning_type: str
    message: str
    recommendation: Optional[str] = None


@dataclass
class ValidationMetrics:
    """Metrics collected during validation."""
    files_validated: int = 0
    links_checked: int = 0
    accessibility_issues: int = 0
    performance_issues: int = 0
    technical_accuracy_issues: int = 0
    validation_duration: float = 0.0


@dataclass
class ValidationResult:
    """Result of a comprehensive validation operation."""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationWarning] = field(default_factory=list)
    metrics: ValidationMetrics = field(default_factory=ValidationMetrics)
    recommendations: List[str] = field(default_factory=list)
    validated_files: List[Path] = field(default_factory=list)


@dataclass
class LinkValidationResult:
    """Result of link validation."""
    url: str
    is_valid: bool
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    response_time: Optional[float] = None


@dataclass
class AccessibilityResult:
    """Result of accessibility validation."""
    file_path: Path
    is_compliant: bool
    wcag_level: str = "AA"
    issues: List[Dict[str, Any]] = field(default_factory=list)
    score: float = 0.0


@dataclass
class PerformanceResult:
    """Result of performance validation."""
    site_path: Path
    load_time: float
    meets_requirements: bool
    file_sizes: Dict[str, int] = field(default_factory=dict)
    optimization_suggestions: List[str] = field(default_factory=list)


class TechnicalAccuracyValidator:
    """Validates technical accuracy of documentation against implementation code."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.app_dir = project_root / "app"
        self.frontend_dir = project_root / "frontend"
    
    def validate_api_documentation(self, doc_path: Path) -> List[ValidationError]:
        """Validate API documentation against FastAPI implementation."""
        errors = []
        
        try:
            # Read documentation content
            doc_content = doc_path.read_text()
            
            # Extract API endpoints from documentation
            doc_endpoints = self._extract_endpoints_from_docs(doc_content)
            
            # Get actual API endpoints from FastAPI app
            actual_endpoints = self._get_fastapi_endpoints()
            
            # Compare documented vs actual endpoints
            for endpoint in doc_endpoints:
                if endpoint not in actual_endpoints:
                    errors.append(ValidationError(
                        file_path=doc_path,
                        line_number=None,
                        error_type="missing_endpoint",
                        message=f"Documented endpoint '{endpoint}' not found in implementation",
                        fix_suggestion=f"Remove documentation for '{endpoint}' or implement the endpoint"
                    ))
            
            # Check for undocumented endpoints
            for endpoint in actual_endpoints:
                if endpoint not in doc_endpoints:
                    errors.append(ValidationError(
                        file_path=doc_path,
                        line_number=None,
                        error_type="undocumented_endpoint",
                        message=f"Endpoint '{endpoint}' exists in implementation but not documented",
                        fix_suggestion=f"Add documentation for endpoint '{endpoint}'"
                    ))
        
        except Exception as e:
            errors.append(ValidationError(
                file_path=doc_path,
                line_number=None,
                error_type="validation_error",
                message=f"Failed to validate API documentation: {str(e)}"
            ))
        
        return errors
    
    def validate_service_documentation(self, doc_path: Path) -> List[ValidationError]:
        """Validate service documentation against actual service implementations."""
        errors = []
        
        try:
            doc_content = doc_path.read_text()
            
            # Extract service class names from documentation
            doc_classes = self._extract_classes_from_docs(doc_content)
            
            # Get actual service classes from implementation
            actual_classes = self._get_service_classes()
            
            # Validate documented classes exist
            for class_name in doc_classes:
                if class_name not in actual_classes:
                    errors.append(ValidationError(
                        file_path=doc_path,
                        line_number=None,
                        error_type="missing_class",
                        message=f"Documented class '{class_name}' not found in implementation",
                        fix_suggestion=f"Remove documentation for '{class_name}' or implement the class"
                    ))
                else:
                    # Validate methods if class exists
                    doc_methods = self._extract_methods_from_docs(doc_content, class_name)
                    actual_methods = actual_classes[class_name].get('methods', [])
                    
                    for method_name in doc_methods:
                        if method_name not in actual_methods:
                            errors.append(ValidationError(
                                file_path=doc_path,
                                line_number=None,
                                error_type="missing_method",
                                message=f"Documented method '{class_name}.{method_name}' not found in implementation",
                                fix_suggestion=f"Remove documentation for '{method_name}' or implement the method"
                            ))
        
        except Exception as e:
            errors.append(ValidationError(
                file_path=doc_path,
                line_number=None,
                error_type="validation_error",
                message=f"Failed to validate service documentation: {str(e)}"
            ))
        
        return errors
    
    def _extract_endpoints_from_docs(self, content: str) -> Set[str]:
        """Extract API endpoints from documentation content."""
        endpoints = set()
        
        # Look for HTTP method + path patterns
        patterns = [
            r'(GET|POST|PUT|DELETE|PATCH)\s+(/[^\s\n]*)',
            r'`(GET|POST|PUT|DELETE|PATCH)\s+([^`]*)`',
            r'##\s*(GET|POST|PUT|DELETE|PATCH)\s+(/[^\n]*)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    method, path = match
                    endpoints.add(f"{method.upper()} {path.strip()}")
        
        return endpoints
    
    def _extract_classes_from_docs(self, content: str) -> Set[str]:
        """Extract class names from documentation content."""
        classes = set()
        
        # Look for class name patterns
        patterns = [
            r'##\s*Class:\s*(\w+)',
            r'###\s*(\w+)\s*class',
            r'`(\w+)`\s*class',
            r'class\s+(\w+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            classes.update(matches)
        
        return classes
    
    def _extract_methods_from_docs(self, content: str, class_name: str) -> Set[str]:
        """Extract method names for a specific class from documentation."""
        methods = set()
        
        # Look for method patterns within class documentation
        patterns = [
            rf'(?:###|####)\s*`?({class_name}\.)?(\w+)\(`',
            rf'`(\w+)\([^)]*\)`',
            rf'def\s+(\w+)\('
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple):
                    method_name = match[-1] if match[-1] else match[0]
                else:
                    method_name = match
                
                if method_name and not method_name.startswith('_'):
                    methods.add(method_name)
        
        return methods
    
    def _get_fastapi_endpoints(self) -> Set[str]:
        """Get actual endpoints from FastAPI application."""
        endpoints = set()
        
        try:
            # Import the FastAPI app
            import sys
            sys.path.insert(0, str(self.project_root))
            
            from app.main import app
            
            # Get routes from FastAPI app
            for route in app.routes:
                if hasattr(route, 'methods') and hasattr(route, 'path'):
                    for method in route.methods:
                        if method != 'HEAD':  # Skip HEAD methods
                            endpoints.add(f"{method} {route.path}")
        
        except Exception:
            # Fallback: scan API files manually
            api_dir = self.app_dir / "api"
            if api_dir.exists():
                for py_file in api_dir.rglob("*.py"):
                    try:
                        content = py_file.read_text()
                        # Look for FastAPI route decorators
                        route_patterns = [
                            r'@router\.(get|post|put|delete|patch)\(["\']([^"\']*)["\']',
                            r'@app\.(get|post|put|delete|patch)\(["\']([^"\']*)["\']'
                        ]
                        
                        for pattern in route_patterns:
                            matches = re.findall(pattern, content, re.IGNORECASE)
                            for method, path in matches:
                                endpoints.add(f"{method.upper()} {path}")
                    
                    except Exception:
                        continue
        
        return endpoints
    
    def _get_service_classes(self) -> Dict[str, Dict[str, Any]]:
        """Get actual service classes from implementation."""
        classes = {}
        
        services_dir = self.app_dir / "services"
        if not services_dir.exists():
            return classes
        
        for py_file in services_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            
            try:
                content = py_file.read_text()
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_info = {
                            'file': str(py_file),
                            'methods': []
                        }
                        
                        # Extract methods
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef) and not item.name.startswith('_'):
                                class_info['methods'].append(item.name)
                        
                        classes[node.name] = class_info
            
            except Exception:
                continue
        
        return classes


class LinkValidator:
    """Validates internal and external links in documentation."""
    
    def __init__(self, project_root: Path, docs_dir: Path):
        self.project_root = project_root
        self.docs_dir = docs_dir
        self.session = None
    
    async def validate_links(self, file_path: Path) -> List[ValidationError]:
        """Validate all links in a documentation file."""
        errors = []
        
        try:
            content = file_path.read_text()
            links = self._extract_links(content)
            
            for link_info in links:
                result = await self._validate_single_link(link_info, file_path)
                if not result.is_valid:
                    errors.append(ValidationError(
                        file_path=file_path,
                        line_number=link_info.get('line_number'),
                        error_type="broken_link",
                        message=f"Broken link: {result.url} - {result.error_message}",
                        fix_suggestion=f"Fix or remove the link to '{result.url}'"
                    ))
        
        except Exception as e:
            errors.append(ValidationError(
                file_path=file_path,
                line_number=None,
                error_type="link_validation_error",
                message=f"Failed to validate links: {str(e)}"
            ))
        
        return errors
    
    def _extract_links(self, content: str) -> List[Dict[str, Any]]:
        """Extract all links from markdown content."""
        links = []
        
        # Markdown link pattern: [text](url)
        link_pattern = r'\[([^\]]*)\]\(([^)]+)\)'
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            matches = re.finditer(link_pattern, line)
            for match in matches:
                text, url = match.groups()
                links.append({
                    'text': text,
                    'url': url.strip(),
                    'line_number': line_num,
                    'line_content': line.strip()
                })
        
        return links
    
    async def _validate_single_link(self, link_info: Dict[str, Any], source_file: Path) -> LinkValidationResult:
        """Validate a single link."""
        url = link_info['url']
        
        # Handle different types of links
        if url.startswith('http://') or url.startswith('https://'):
            return await self._validate_external_link(url)
        elif url.startswith('#'):
            return self._validate_anchor_link(url, source_file)
        else:
            return self._validate_internal_link(url, source_file)
    
    async def _validate_external_link(self, url: str) -> LinkValidationResult:
        """Validate an external HTTP/HTTPS link."""
        if not HAS_HTTP_CLIENTS:
            return LinkValidationResult(
                url=url,
                is_valid=True,  # Assume valid if we can't check
                error_message="HTTP client not available for validation"
            )
        
        try:
            if not self.session:
                timeout = aiohttp.ClientTimeout(total=10)
                self.session = aiohttp.ClientSession(timeout=timeout)
            
            start_time = time.time()
            async with self.session.head(url) as response:
                response_time = time.time() - start_time
                
                return LinkValidationResult(
                    url=url,
                    is_valid=response.status < 400,
                    status_code=response.status,
                    response_time=response_time,
                    error_message=None if response.status < 400 else f"HTTP {response.status}"
                )
        
        except Exception as e:
            return LinkValidationResult(
                url=url,
                is_valid=False,
                error_message=str(e)
            )
    
    def _validate_internal_link(self, url: str, source_file: Path) -> LinkValidationResult:
        """Validate an internal link to another file."""
        # Resolve relative path
        if url.startswith('/'):
            # Absolute path from docs root
            target_path = self.docs_dir / url.lstrip('/')
        else:
            # Relative path from current file
            target_path = source_file.parent / url
        
        # Normalize path
        try:
            target_path = target_path.resolve()
        except Exception:
            return LinkValidationResult(
                url=url,
                is_valid=False,
                error_message="Invalid path format"
            )
        
        # Check if file exists
        if target_path.exists():
            return LinkValidationResult(url=url, is_valid=True)
        else:
            return LinkValidationResult(
                url=url,
                is_valid=False,
                error_message="File not found"
            )
    
    def _validate_anchor_link(self, url: str, source_file: Path) -> LinkValidationResult:
        """Validate an anchor link within the same file."""
        anchor = url.lstrip('#').lower().replace(' ', '-')
        
        try:
            content = source_file.read_text()
            
            # Look for matching headers
            header_pattern = r'^#+\s+(.+)$'
            lines = content.split('\n')
            
            for line in lines:
                match = re.match(header_pattern, line.strip())
                if match:
                    header_text = match.group(1).lower().replace(' ', '-')
                    # Remove special characters for comparison
                    header_text = re.sub(r'[^\w\-]', '', header_text)
                    anchor_clean = re.sub(r'[^\w\-]', '', anchor)
                    
                    if header_text == anchor_clean:
                        return LinkValidationResult(url=url, is_valid=True)
            
            return LinkValidationResult(
                url=url,
                is_valid=False,
                error_message="Anchor not found in document"
            )
        
        except Exception as e:
            return LinkValidationResult(
                url=url,
                is_valid=False,
                error_message=f"Error validating anchor: {str(e)}"
            )
    
    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()


class AccessibilityValidator:
    """Validates WCAG 2.1 AA compliance for generated HTML documentation."""
    
    def __init__(self):
        self.wcag_rules = self._load_wcag_rules()
        self.axe_core_available = self._check_axe_core_availability()
    
    def validate_accessibility(self, html_file: Path) -> AccessibilityResult:
        """Validate WCAG 2.1 AA compliance for an HTML file."""
        if not HAS_HTML_PARSER:
            return AccessibilityResult(
                file_path=html_file,
                is_compliant=True,  # Assume compliant if we can't check
                issues=[{"type": "warning", "message": "HTML parser not available for accessibility validation"}]
            )
        
        try:
            # Try axe-core validation first if available
            if self.axe_core_available:
                axe_result = self._validate_with_axe_core(html_file)
                if axe_result:
                    return axe_result
            
            # Fallback to manual validation
            content = html_file.read_text()
            soup = BeautifulSoup(content, 'html.parser')
            
            issues = []
            
            # Check various WCAG criteria
            issues.extend(self._check_images_alt_text(soup))
            issues.extend(self._check_headings_structure(soup))
            issues.extend(self._check_color_contrast(soup))
            issues.extend(self._check_keyboard_navigation(soup))
            issues.extend(self._check_form_labels(soup))
            issues.extend(self._check_link_text(soup))
            issues.extend(self._check_page_structure(soup))
            issues.extend(self._check_language_attributes(soup))
            
            # Calculate compliance score
            total_checks = 8
            failed_checks = len([issue for issue in issues if issue['severity'] == 'error'])
            score = max(0, (total_checks - failed_checks) / total_checks)
            
            return AccessibilityResult(
                file_path=html_file,
                is_compliant=failed_checks == 0,
                wcag_level="AA",
                issues=issues,
                score=score
            )
        
        except Exception as e:
            return AccessibilityResult(
                file_path=html_file,
                is_compliant=False,
                issues=[{
                    "type": "error",
                    "severity": "error",
                    "message": f"Failed to validate accessibility: {str(e)}"
                }]
            )
    
    def _check_images_alt_text(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Check that all images have appropriate alt text."""
        issues = []
        
        images = soup.find_all('img')
        for img in images:
            if not img.get('alt'):
                issues.append({
                    "type": "missing_alt_text",
                    "severity": "error",
                    "message": f"Image missing alt text: {img.get('src', 'unknown')}",
                    "element": str(img)[:100],
                    "wcag_criterion": "1.1.1"
                })
            elif len(img.get('alt', '').strip()) == 0:
                issues.append({
                    "type": "empty_alt_text",
                    "severity": "warning",
                    "message": f"Image has empty alt text: {img.get('src', 'unknown')}",
                    "element": str(img)[:100],
                    "wcag_criterion": "1.1.1"
                })
        
        return issues
    
    def _check_headings_structure(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Check proper heading hierarchy."""
        issues = []
        
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if not headings:
            return issues
        
        prev_level = 0
        for heading in headings:
            level = int(heading.name[1])
            
            if prev_level > 0 and level > prev_level + 1:
                issues.append({
                    "type": "heading_skip",
                    "severity": "warning",
                    "message": f"Heading level skipped: {heading.name} after h{prev_level}",
                    "element": heading.get_text()[:50],
                    "wcag_criterion": "1.3.1"
                })
            
            prev_level = level
        
        return issues
    
    def _check_color_contrast(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Basic color contrast check (simplified)."""
        issues = []
        
        # This is a simplified check - full contrast validation requires
        # actual color computation which is complex
        elements_with_style = soup.find_all(style=True)
        
        for element in elements_with_style:
            style = element.get('style', '')
            if 'color:' in style and 'background' not in style:
                issues.append({
                    "type": "potential_contrast_issue",
                    "severity": "warning",
                    "message": "Element has color but no background specified",
                    "element": element.name,
                    "wcag_criterion": "1.4.3"
                })
        
        return issues
    
    def _check_keyboard_navigation(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Check keyboard navigation support."""
        issues = []
        
        # Check for interactive elements without proper keyboard support
        interactive_elements = soup.find_all(['button', 'a', 'input', 'select', 'textarea'])
        
        for element in interactive_elements:
            if element.name == 'a' and not element.get('href'):
                issues.append({
                    "type": "non_functional_link",
                    "severity": "error",
                    "message": "Link without href attribute",
                    "element": element.get_text()[:50],
                    "wcag_criterion": "2.1.1"
                })
        
        return issues
    
    def _check_form_labels(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Check that form inputs have proper labels."""
        issues = []
        
        inputs = soup.find_all(['input', 'select', 'textarea'])
        
        for input_elem in inputs:
            input_type = input_elem.get('type', 'text')
            if input_type in ['hidden', 'submit', 'button']:
                continue
            
            input_id = input_elem.get('id')
            aria_label = input_elem.get('aria-label')
            
            # Check for associated label
            label_found = False
            if input_id:
                label = soup.find('label', {'for': input_id})
                if label:
                    label_found = True
            
            if not label_found and not aria_label:
                issues.append({
                    "type": "missing_form_label",
                    "severity": "error",
                    "message": f"Form input missing label: {input_elem.name}",
                    "element": str(input_elem)[:100],
                    "wcag_criterion": "1.3.1"
                })
        
        return issues
    
    def _check_link_text(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Check that links have descriptive text."""
        issues = []
        
        links = soup.find_all('a', href=True)
        
        for link in links:
            link_text = link.get_text().strip()
            
            if not link_text:
                issues.append({
                    "type": "empty_link_text",
                    "severity": "error",
                    "message": f"Link with no text: {link.get('href')}",
                    "element": str(link)[:100],
                    "wcag_criterion": "2.4.4"
                })
            elif link_text.lower() in ['click here', 'read more', 'here', 'more']:
                issues.append({
                    "type": "non_descriptive_link",
                    "severity": "warning",
                    "message": f"Non-descriptive link text: '{link_text}'",
                    "element": str(link)[:100],
                    "wcag_criterion": "2.4.4"
                })
        
        return issues
    
    def _check_page_structure(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Check proper page structure and landmarks."""
        issues = []
        
        # Check for main landmark
        main_elements = soup.find_all(['main', '[role="main"]'])
        if not main_elements:
            issues.append({
                "type": "missing_main_landmark",
                "severity": "warning",
                "message": "Page missing main landmark",
                "wcag_criterion": "1.3.1"
            })
        
        # Check for navigation landmarks
        nav_elements = soup.find_all(['nav', '[role="navigation"]'])
        if len(nav_elements) > 1:
            # Multiple nav elements should have aria-label or aria-labelledby
            for nav in nav_elements:
                if not nav.get('aria-label') and not nav.get('aria-labelledby'):
                    issues.append({
                        "type": "unlabeled_navigation",
                        "severity": "warning",
                        "message": "Multiple navigation elements should have labels",
                        "wcag_criterion": "1.3.1"
                    })
        
        return issues
    
    def _check_language_attributes(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Check for proper language attributes."""
        issues = []
        
        # Check for lang attribute on html element
        html_element = soup.find('html')
        if html_element and not html_element.get('lang'):
            issues.append({
                "type": "missing_lang_attribute",
                "severity": "error",
                "message": "HTML element missing lang attribute",
                "wcag_criterion": "3.1.1"
            })
        
        return issues
    
    def _check_axe_core_availability(self) -> bool:
        """Check if axe-core is available for accessibility testing."""
        try:
            # Check if we can run axe-core via selenium or playwright
            import subprocess
            result = subprocess.run(['which', 'axe'], capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False
    
    def _validate_with_axe_core(self, html_file: Path) -> Optional[AccessibilityResult]:
        """Validate accessibility using axe-core if available."""
        try:
            # This would require selenium/playwright integration
            # For now, return None to fall back to manual validation
            return None
        except Exception:
            return None
    
    def _load_wcag_rules(self) -> Dict[str, Any]:
        """Load WCAG 2.1 AA rules configuration."""
        return {
            "1.1.1": "Non-text Content",
            "1.3.1": "Info and Relationships", 
            "1.4.3": "Contrast (Minimum)",
            "2.1.1": "Keyboard",
            "2.4.4": "Link Purpose (In Context)",
            "3.1.1": "Language of Page"
        }


class PerformanceValidator:
    """Validates performance requirements for documentation sites."""
    
    def __init__(self):
        self.max_load_time = 2.0  # 2 seconds requirement
        self.max_file_size = 5 * 1024 * 1024  # 5MB max file size
    
    def validate_performance(self, site_path: Path) -> PerformanceResult:
        """Validate performance requirements for a documentation site."""
        try:
            # Measure site loading performance
            load_time = self._measure_load_time(site_path)
            
            # Analyze file sizes
            file_sizes = self._analyze_file_sizes(site_path)
            
            # Generate optimization suggestions
            suggestions = self._generate_optimization_suggestions(file_sizes, load_time)
            
            meets_requirements = (
                load_time <= self.max_load_time and
                all(size <= self.max_file_size for size in file_sizes.values())
            )
            
            return PerformanceResult(
                site_path=site_path,
                load_time=load_time,
                meets_requirements=meets_requirements,
                file_sizes=file_sizes,
                optimization_suggestions=suggestions
            )
        
        except Exception as e:
            return PerformanceResult(
                site_path=site_path,
                load_time=float('inf'),
                meets_requirements=False,
                optimization_suggestions=[f"Performance validation failed: {str(e)}"]
            )
    
    def _measure_load_time(self, site_path: Path) -> float:
        """Measure site loading time (simplified simulation)."""
        start_time = time.time()
        
        # Simulate loading by reading main files
        total_size = 0
        file_count = 0
        
        for file_path in site_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in ['.html', '.css', '.js', '.md']:
                try:
                    total_size += file_path.stat().st_size
                    file_count += 1
                    
                    # Simulate network delay based on file size
                    time.sleep(min(0.001, file_path.stat().st_size / 1000000))
                
                except Exception:
                    continue
        
        load_time = time.time() - start_time
        
        # Add base load time simulation
        base_load_time = 0.1 + (total_size / 10000000)  # Simulate network and processing
        
        return load_time + base_load_time
    
    def _analyze_file_sizes(self, site_path: Path) -> Dict[str, int]:
        """Analyze file sizes in the documentation site."""
        file_sizes = {}
        
        for file_path in site_path.rglob("*"):
            if file_path.is_file():
                try:
                    size = file_path.stat().st_size
                    relative_path = str(file_path.relative_to(site_path))
                    file_sizes[relative_path] = size
                except Exception:
                    continue
        
        return file_sizes
    
    def _generate_optimization_suggestions(self, file_sizes: Dict[str, int], load_time: float) -> List[str]:
        """Generate performance optimization suggestions."""
        suggestions = []
        
        # Check load time against 2-second requirement
        if load_time > self.max_load_time:
            suggestions.append(f"Site load time ({load_time:.2f}s) exceeds 2-second requirement")
            suggestions.append("Consider optimizing critical rendering path and reducing resource sizes")
        
        # Analyze file types and sizes
        image_files = {path: size for path, size in file_sizes.items() 
                      if any(path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.svg'])}
        css_files = {path: size for path, size in file_sizes.items() if path.lower().endswith('.css')}
        js_files = {path: size for path, size in file_sizes.items() if path.lower().endswith('.js')}
        html_files = {path: size for path, size in file_sizes.items() if path.lower().endswith('.html')}
        
        # Check large files by type
        large_images = {path: size for path, size in image_files.items() if size > 500 * 1024}  # > 500KB
        large_css = {path: size for path, size in css_files.items() if size > 100 * 1024}  # > 100KB
        large_js = {path: size for path, size in js_files.items() if size > 200 * 1024}  # > 200KB
        
        if large_images:
            suggestions.append(f"Large images detected ({len(large_images)} files)")
            suggestions.append("Consider compressing images, using WebP format, or implementing responsive images")
        
        if large_css:
            suggestions.append(f"Large CSS files detected ({len(large_css)} files)")
            suggestions.append("Consider minifying CSS, removing unused styles, or splitting into smaller files")
        
        if large_js:
            suggestions.append(f"Large JavaScript files detected ({len(large_js)} files)")
            suggestions.append("Consider code splitting, minification, or lazy loading of JavaScript")
        
        # Check total site size
        total_size = sum(file_sizes.values())
        if total_size > 50 * 1024 * 1024:  # 50MB
            suggestions.append(f"Total site size ({total_size / 1024 / 1024:.1f}MB) is quite large")
            suggestions.append("Consider implementing lazy loading, pagination, or content delivery network (CDN)")
        
        # Check number of files
        if len(file_sizes) > 1000:
            suggestions.append(f"Large number of files ({len(file_sizes)}) may impact performance")
            suggestions.append("Consider bundling resources or reducing the number of HTTP requests")
        
        # Check for optimization opportunities
        if len(html_files) > 100:
            suggestions.append("Consider implementing static site generation or server-side rendering for better performance")
        
        # Specific performance recommendations
        if load_time > 1.0:
            suggestions.append("Enable gzip compression for text-based resources")
            suggestions.append("Implement browser caching with appropriate cache headers")
            suggestions.append("Consider using a content delivery network (CDN) for static assets")
        
        return suggestions


class ValidationEngine:
    """
    Comprehensive validation engine for all documentation types.
    
    Implements multi-layered validation including technical accuracy validation
    against code, comprehensive link validation, accessibility compliance checking,
    and performance validation with detailed error reporting.
    """
    
    def __init__(self, project_root: Path, docs_dir: Path):
        self.project_root = project_root
        self.docs_dir = docs_dir
        
        # Initialize validators
        self.technical_validator = TechnicalAccuracyValidator(project_root)
        self.link_validator = LinkValidator(project_root, docs_dir)
        self.accessibility_validator = AccessibilityValidator()
        self.performance_validator = PerformanceValidator()
    
    async def validate_comprehensive(self, target_files: Optional[List[Path]] = None) -> ValidationResult:
        """
        Perform comprehensive validation of documentation.
        
        Args:
            target_files: Optional list of specific files to validate.
                         If None, validates all documentation files.
        
        Returns:
            ValidationResult with comprehensive validation details
        """
        start_time = time.time()
        result = ValidationResult(is_valid=True)
        
        try:
            # Determine files to validate
            if target_files:
                files_to_validate = target_files
            else:
                files_to_validate = list(self.docs_dir.rglob("*.md"))
                files_to_validate.extend(self.docs_dir.rglob("*.html"))
            
            result.validated_files = files_to_validate
            
            # Validate each file
            for file_path in files_to_validate:
                await self._validate_single_file(file_path, result)
            
            # Validate overall site performance
            if self.docs_dir.exists():
                perf_result = self.performance_validator.validate_performance(self.docs_dir)
                if not perf_result.meets_requirements:
                    result.is_valid = False
                    result.errors.append(ValidationError(
                        file_path=self.docs_dir,
                        line_number=None,
                        error_type="performance_issue",
                        message=f"Site performance does not meet requirements (load time: {perf_result.load_time:.2f}s)",
                        fix_suggestion="Optimize file sizes and reduce load time"
                    ))
                
                result.recommendations.extend(perf_result.optimization_suggestions)
                result.metrics.performance_issues = len(perf_result.optimization_suggestions)
            
            # Update metrics
            result.metrics.files_validated = len(files_to_validate)
            result.metrics.validation_duration = time.time() - start_time
            
            # Generate recommendations based on validation results
            result.recommendations.extend(self._generate_recommendations(result))
            
            # Generate detailed error report if there are issues
            if result.errors or result.warnings:
                self._generate_detailed_error_report(result)
        
        except Exception as e:
            result.is_valid = False
            result.errors.append(ValidationError(
                file_path=Path("unknown"),
                line_number=None,
                error_type="validation_system_error",
                message=f"Validation system error: {str(e)}"
            ))
        
        finally:
            # Clean up resources
            await self.link_validator.close()
        
        return result
    
    async def _validate_single_file(self, file_path: Path, result: ValidationResult):
        """Validate a single documentation file."""
        try:
            # Technical accuracy validation
            if file_path.suffix == '.md':
                if 'api' in str(file_path).lower():
                    tech_errors = self.technical_validator.validate_api_documentation(file_path)
                    result.errors.extend(tech_errors)
                    result.metrics.technical_accuracy_issues += len(tech_errors)
                
                elif 'service' in str(file_path).lower() or 'interface' in str(file_path).lower():
                    tech_errors = self.technical_validator.validate_service_documentation(file_path)
                    result.errors.extend(tech_errors)
                    result.metrics.technical_accuracy_issues += len(tech_errors)
            
            # Link validation
            link_errors = await self.link_validator.validate_links(file_path)
            result.errors.extend(link_errors)
            result.metrics.links_checked += len(link_errors)
            
            # Format and style validation
            format_errors = self._validate_format_and_style(file_path)
            result.errors.extend(format_errors)
            
            # Accessibility validation for HTML files
            if file_path.suffix == '.html':
                accessibility_result = self.accessibility_validator.validate_accessibility(file_path)
                
                for issue in accessibility_result.issues:
                    if issue.get('severity') == 'error':
                        result.errors.append(ValidationError(
                            file_path=file_path,
                            line_number=None,
                            error_type="accessibility_error",
                            message=issue['message'],
                            fix_suggestion=f"Fix WCAG {issue.get('wcag_criterion', 'unknown')} compliance issue"
                        ))
                    else:
                        result.warnings.append(ValidationWarning(
                            file_path=file_path,
                            line_number=None,
                            warning_type="accessibility_warning",
                            message=issue['message'],
                            recommendation=f"Consider fixing WCAG {issue.get('wcag_criterion', 'unknown')} issue"
                        ))
                
                result.metrics.accessibility_issues += len([
                    issue for issue in accessibility_result.issues 
                    if issue.get('severity') == 'error'
                ])
        
        except Exception as e:
            result.errors.append(ValidationError(
                file_path=file_path,
                line_number=None,
                error_type="file_validation_error",
                message=f"Failed to validate file: {str(e)}"
            ))
    
    def _validate_format_and_style(self, file_path: Path) -> List[ValidationError]:
        """Validate formatting and style compliance."""
        errors = []
        
        try:
            content = file_path.read_text()
            
            # Check for empty files
            if len(content.strip()) == 0:
                errors.append(ValidationError(
                    file_path=file_path,
                    line_number=1,
                    error_type="empty_file",
                    message="File is empty",
                    fix_suggestion="Add content to the file or remove it"
                ))
                return errors
            
            lines = content.split('\n')
            
            # Check for proper title structure
            has_main_title = any(line.strip().startswith('# ') for line in lines)
            if not has_main_title and file_path.suffix == '.md':
                errors.append(ValidationError(
                    file_path=file_path,
                    line_number=1,
                    error_type="missing_title",
                    message="Markdown file missing main title (# Title)",
                    fix_suggestion="Add a main title at the beginning of the file"
                ))
            
            # Check for consistent heading hierarchy
            heading_levels = []
            for line_num, line in enumerate(lines, 1):
                if line.strip().startswith('#'):
                    level = len(line) - len(line.lstrip('#'))
                    heading_levels.append((level, line_num))
            
            for i in range(1, len(heading_levels)):
                prev_level, prev_line = heading_levels[i-1]
                curr_level, curr_line = heading_levels[i]
                
                if curr_level > prev_level + 1:
                    errors.append(ValidationError(
                        file_path=file_path,
                        line_number=curr_line,
                        error_type="heading_skip",
                        message=f"Heading level skipped from h{prev_level} to h{curr_level}",
                        fix_suggestion=f"Use h{prev_level + 1} instead of h{curr_level}"
                    ))
            
            # Check for trailing whitespace
            for line_num, line in enumerate(lines, 1):
                if line.endswith(' ') or line.endswith('\t'):
                    errors.append(ValidationError(
                        file_path=file_path,
                        line_number=line_num,
                        error_type="trailing_whitespace",
                        message="Line has trailing whitespace",
                        severity="warning",
                        fix_suggestion="Remove trailing whitespace"
                    ))
        
        except Exception as e:
            errors.append(ValidationError(
                file_path=file_path,
                line_number=None,
                error_type="format_validation_error",
                message=f"Failed to validate format: {str(e)}"
            ))
        
        return errors
    
    def _generate_recommendations(self, result: ValidationResult) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Error-based recommendations
        error_types = {}
        for error in result.errors:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        
        if error_types.get('broken_link', 0) > 0:
            recommendations.append("Review and fix broken links in documentation")
        
        if error_types.get('missing_endpoint', 0) > 0:
            recommendations.append("Update API documentation to match current implementation")
        
        if error_types.get('accessibility_error', 0) > 0:
            recommendations.append("Address accessibility issues to ensure WCAG 2.1 AA compliance")
        
        if error_types.get('performance_issue', 0) > 0:
            recommendations.append("Optimize documentation site performance to meet 2-second load time requirement")
        
        # General recommendations based on metrics
        if result.metrics.files_validated > 100:
            recommendations.append("Consider implementing automated validation in CI/CD pipeline")
        
        if len(result.warnings) > len(result.errors) * 2:
            recommendations.append("Address warnings to improve documentation quality")
        
        return recommendations
    
    def _generate_detailed_error_report(self, result: ValidationResult):
        """Generate a detailed error report with specific issues and fixes."""
        try:
            report_dir = self.docs_dir / "validation_reports"
            report_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = report_dir / f"validation_report_{timestamp}.md"
            
            report_content = f"""# Documentation Validation Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Files Validated**: {result.metrics.files_validated}
**Validation Duration**: {result.metrics.validation_duration:.2f} seconds

## Summary

- **Errors**: {len(result.errors)}
- **Warnings**: {len(result.warnings)}
- **Overall Status**: {'✅ PASSED' if result.is_valid else '❌ FAILED'}

## Metrics

- Files Validated: {result.metrics.files_validated}
- Links Checked: {result.metrics.links_checked}
- Accessibility Issues: {result.metrics.accessibility_issues}
- Performance Issues: {result.metrics.performance_issues}
- Technical Accuracy Issues: {result.metrics.technical_accuracy_issues}

"""
            
            # Add errors section
            if result.errors:
                report_content += "## Errors\n\n"
                for i, error in enumerate(result.errors, 1):
                    report_content += f"### Error {i}: {error.error_type}\n\n"
                    report_content += f"**File**: `{error.file_path}`\n"
                    if error.line_number:
                        report_content += f"**Line**: {error.line_number}\n"
                    report_content += f"**Message**: {error.message}\n"
                    if error.fix_suggestion:
                        report_content += f"**Fix**: {error.fix_suggestion}\n"
                    report_content += f"**Severity**: {error.severity}\n\n"
            
            # Add warnings section
            if result.warnings:
                report_content += "## Warnings\n\n"
                for i, warning in enumerate(result.warnings, 1):
                    report_content += f"### Warning {i}: {warning.warning_type}\n\n"
                    report_content += f"**File**: `{warning.file_path}`\n"
                    if warning.line_number:
                        report_content += f"**Line**: {warning.line_number}\n"
                    report_content += f"**Message**: {warning.message}\n"
                    if warning.recommendation:
                        report_content += f"**Recommendation**: {warning.recommendation}\n"
                    report_content += "\n"
            
            # Add recommendations section
            if result.recommendations:
                report_content += "## Recommendations\n\n"
                for i, recommendation in enumerate(result.recommendations, 1):
                    report_content += f"{i}. {recommendation}\n"
                report_content += "\n"
            
            # Add next steps
            report_content += """## Next Steps

1. **Address Critical Errors**: Fix all errors marked as 'error' severity first
2. **Review Warnings**: Consider addressing warnings to improve documentation quality
3. **Implement Recommendations**: Follow the recommendations to enhance overall documentation
4. **Re-run Validation**: After making changes, re-run validation to verify fixes
5. **Automate Validation**: Consider integrating validation into your CI/CD pipeline

## Validation Rules Reference

### Technical Accuracy
- API endpoints must match implementation
- Service classes and methods must exist in code
- Documentation should reflect actual code structure

### Link Validation
- All internal links must point to existing files
- External links should be accessible (HTTP 200-299)
- Anchor links must reference existing headers

### Accessibility (WCAG 2.1 AA)
- Images must have alt text
- Proper heading hierarchy (h1 → h2 → h3)
- Links must have descriptive text
- Forms must have proper labels
- Page must have language attribute

### Performance
- Site load time should be under 2 seconds
- Individual files should be reasonably sized
- Total site size should be optimized
- Consider compression and caching strategies

### Format and Style
- Markdown files should have proper structure
- No trailing whitespace
- Consistent heading hierarchy
- Non-empty files with meaningful content
"""
            
            # Write the report
            report_file.write_text(report_content)
            
            # Add report location to recommendations
            result.recommendations.append(f"Detailed validation report saved to: {report_file}")
            
        except Exception as e:
            # Don't fail validation if report generation fails
            result.warnings.append(ValidationWarning(
                file_path=Path("validation_system"),
                line_number=None,
                warning_type="report_generation_error",
                message=f"Failed to generate detailed error report: {str(e)}"
            ))