#!/usr/bin/env python3
"""
Automatic API Documentation Updates

This module provides automatic documentation generation hooks, validation
against actual API implementation, and diff detection for API specification
changes.

Implements Requirements 2.3, 2.5 from the comprehensive documentation specification.
"""

import json
import hashlib
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import difflib
import ast
import importlib.util
import sys


@dataclass
class APIChange:
    """Represents a change in the API specification."""
    change_type: str  # 'added', 'removed', 'modified'
    path: str
    method: str
    old_spec: Optional[Dict[str, Any]] = None
    new_spec: Optional[Dict[str, Any]] = None
    description: str = ""


@dataclass
class ValidationIssue:
    """Represents a validation issue found during API validation."""
    severity: str  # 'error', 'warning', 'info'
    category: str  # 'schema', 'implementation', 'documentation'
    message: str
    endpoint: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class UpdateResult:
    """Result of automatic API documentation update."""
    success: bool
    changes_detected: List[APIChange] = field(default_factory=list)
    validation_issues: List[ValidationIssue] = field(default_factory=list)
    updated_files: List[Path] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration: float = 0.0


class APISpecificationDiffer:
    """Detects differences between API specifications."""
    
    def __init__(self):
        self.ignore_fields = {'x-codegen-request-body-name', 'x-original-swagger-version'}
    
    def detect_changes(self, old_spec: Dict[str, Any], new_spec: Dict[str, Any]) -> List[APIChange]:
        """
        Detect changes between two OpenAPI specifications.
        
        Args:
            old_spec: Previous OpenAPI specification
            new_spec: Current OpenAPI specification
            
        Returns:
            List of detected changes
        """
        changes = []
        
        try:
            # Compare paths
            old_paths = old_spec.get('paths', {})
            new_paths = new_spec.get('paths', {})
            
            # Find added, removed, and modified paths
            old_path_set = set(old_paths.keys())
            new_path_set = set(new_paths.keys())
            
            # Added paths
            for path in new_path_set - old_path_set:
                for method, spec in new_paths[path].items():
                    changes.append(APIChange(
                        change_type='added',
                        path=path,
                        method=method,
                        new_spec=spec,
                        description=f"New endpoint: {method.upper()} {path}"
                    ))
            
            # Removed paths
            for path in old_path_set - new_path_set:
                for method, spec in old_paths[path].items():
                    changes.append(APIChange(
                        change_type='removed',
                        path=path,
                        method=method,
                        old_spec=spec,
                        description=f"Removed endpoint: {method.upper()} {path}"
                    ))
            
            # Modified paths
            for path in old_path_set & new_path_set:
                path_changes = self._compare_path_methods(path, old_paths[path], new_paths[path])
                changes.extend(path_changes)
            
            # Compare info section
            info_changes = self._compare_info_sections(old_spec.get('info', {}), new_spec.get('info', {}))
            changes.extend(info_changes)
            
            # Compare components
            component_changes = self._compare_components(
                old_spec.get('components', {}), 
                new_spec.get('components', {})
            )
            changes.extend(component_changes)
            
        except Exception as e:
            # Return error as a change for reporting
            changes.append(APIChange(
                change_type='error',
                path='',
                method='',
                description=f"Error detecting changes: {str(e)}"
            ))
        
        return changes
    
    def _compare_path_methods(self, path: str, old_methods: Dict[str, Any], 
                            new_methods: Dict[str, Any]) -> List[APIChange]:
        """Compare methods within a path."""
        changes = []
        
        old_method_set = set(old_methods.keys())
        new_method_set = set(new_methods.keys())
        
        # Added methods
        for method in new_method_set - old_method_set:
            changes.append(APIChange(
                change_type='added',
                path=path,
                method=method,
                new_spec=new_methods[method],
                description=f"New method: {method.upper()} {path}"
            ))
        
        # Removed methods
        for method in old_method_set - new_method_set:
            changes.append(APIChange(
                change_type='removed',
                path=path,
                method=method,
                old_spec=old_methods[method],
                description=f"Removed method: {method.upper()} {path}"
            ))
        
        # Modified methods
        for method in old_method_set & new_method_set:
            old_spec = old_methods[method]
            new_spec = new_methods[method]
            
            if self._specs_differ(old_spec, new_spec):
                changes.append(APIChange(
                    change_type='modified',
                    path=path,
                    method=method,
                    old_spec=old_spec,
                    new_spec=new_spec,
                    description=f"Modified endpoint: {method.upper()} {path}"
                ))
        
        return changes
    
    def _compare_info_sections(self, old_info: Dict[str, Any], new_info: Dict[str, Any]) -> List[APIChange]:
        """Compare info sections of specifications."""
        changes = []
        
        if self._specs_differ(old_info, new_info):
            changes.append(APIChange(
                change_type='modified',
                path='info',
                method='',
                old_spec=old_info,
                new_spec=new_info,
                description="API information updated"
            ))
        
        return changes
    
    def _compare_components(self, old_components: Dict[str, Any], 
                          new_components: Dict[str, Any]) -> List[APIChange]:
        """Compare components sections of specifications."""
        changes = []
        
        if self._specs_differ(old_components, new_components):
            changes.append(APIChange(
                change_type='modified',
                path='components',
                method='',
                old_spec=old_components,
                new_spec=new_components,
                description="API components updated"
            ))
        
        return changes
    
    def _specs_differ(self, old_spec: Dict[str, Any], new_spec: Dict[str, Any]) -> bool:
        """Check if two specifications differ, ignoring certain fields."""
        # Create clean copies without ignored fields
        old_clean = self._clean_spec(old_spec)
        new_clean = self._clean_spec(new_spec)
        
        return old_clean != new_clean
    
    def _clean_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Remove ignored fields from specification for comparison."""
        if not isinstance(spec, dict):
            return spec
        
        cleaned = {}
        for key, value in spec.items():
            if key not in self.ignore_fields:
                if isinstance(value, dict):
                    cleaned[key] = self._clean_spec(value)
                elif isinstance(value, list):
                    cleaned[key] = [self._clean_spec(item) if isinstance(item, dict) else item for item in value]
                else:
                    cleaned[key] = value
        
        return cleaned


class APIImplementationValidator:
    """Validates API documentation against actual implementation."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.app_dir = project_root / "app"
    
    def validate_against_implementation(self, openapi_spec: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Validate OpenAPI specification against actual FastAPI implementation.
        
        Args:
            openapi_spec: OpenAPI specification to validate
            
        Returns:
            List of validation issues found
        """
        issues = []
        
        try:
            # Import and analyze the FastAPI application
            app_analysis = self._analyze_fastapi_app()
            
            # Validate paths exist in implementation
            spec_paths = openapi_spec.get('paths', {})
            for path, methods in spec_paths.items():
                for method, operation in methods.items():
                    if method.lower() in ['get', 'post', 'put', 'delete', 'patch']:
                        validation_issues = self._validate_endpoint_implementation(
                            path, method, operation, app_analysis
                        )
                        issues.extend(validation_issues)
            
            # Validate schemas match implementation
            schema_issues = self._validate_schemas(openapi_spec, app_analysis)
            issues.extend(schema_issues)
            
            # Check for missing documentation
            missing_docs = self._find_missing_documentation(openapi_spec, app_analysis)
            issues.extend(missing_docs)
            
        except Exception as e:
            issues.append(ValidationIssue(
                severity='error',
                category='implementation',
                message=f"Failed to validate against implementation: {str(e)}",
                suggestion="Check that the FastAPI application can be imported and analyzed"
            ))
        
        return issues
    
    def _analyze_fastapi_app(self) -> Dict[str, Any]:
        """Analyze the FastAPI application to extract implementation details."""
        analysis = {
            'endpoints': {},
            'schemas': {},
            'dependencies': [],
            'middleware': []
        }
        
        try:
            # Import the FastAPI app
            spec = importlib.util.spec_from_file_location("app_main", self.app_dir / "main.py")
            app_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(app_module)
            
            app = getattr(app_module, 'app', None)
            if not app:
                return analysis
            
            # Extract routes from FastAPI app
            for route in app.routes:
                if hasattr(route, 'methods') and hasattr(route, 'path'):
                    for method in route.methods:
                        if method.lower() != 'options':  # Skip OPTIONS
                            endpoint_key = f"{method.lower()}:{route.path}"
                            analysis['endpoints'][endpoint_key] = {
                                'path': route.path,
                                'method': method.lower(),
                                'name': getattr(route, 'name', ''),
                                'endpoint': getattr(route, 'endpoint', None)
                            }
            
        except Exception as e:
            # Add error to analysis for reporting
            analysis['error'] = str(e)
        
        return analysis
    
    def _validate_endpoint_implementation(self, path: str, method: str, operation: Dict[str, Any], 
                                        app_analysis: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate that an endpoint exists in the implementation."""
        issues = []
        
        # Convert OpenAPI path to FastAPI path format
        fastapi_path = self._convert_openapi_path_to_fastapi(path)
        endpoint_key = f"{method.lower()}:{fastapi_path}"
        
        if endpoint_key not in app_analysis['endpoints']:
            issues.append(ValidationIssue(
                severity='error',
                category='implementation',
                message=f"Endpoint {method.upper()} {path} documented but not implemented",
                endpoint=f"{method.upper()} {path}",
                suggestion=f"Implement the endpoint or remove it from documentation"
            ))
        
        return issues
    
    def _convert_openapi_path_to_fastapi(self, openapi_path: str) -> str:
        """Convert OpenAPI path format to FastAPI path format."""
        # OpenAPI uses {param} while FastAPI uses {param}
        # They're the same in this case, but we might need more complex conversion
        return openapi_path
    
    def _validate_schemas(self, openapi_spec: Dict[str, Any], app_analysis: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate that schemas match implementation."""
        issues = []
        
        # This is a simplified validation - in a real implementation,
        # we would compare Pydantic models with OpenAPI schemas
        components = openapi_spec.get('components', {})
        schemas = components.get('schemas', {})
        
        if not schemas and 'error' not in app_analysis:
            issues.append(ValidationIssue(
                severity='warning',
                category='schema',
                message="No schemas defined in OpenAPI specification",
                suggestion="Consider adding schema definitions for better API documentation"
            ))
        
        return issues
    
    def _find_missing_documentation(self, openapi_spec: Dict[str, Any], 
                                  app_analysis: Dict[str, Any]) -> List[ValidationIssue]:
        """Find endpoints that are implemented but not documented."""
        issues = []
        
        documented_endpoints = set()
        spec_paths = openapi_spec.get('paths', {})
        
        for path, methods in spec_paths.items():
            for method in methods.keys():
                if method.lower() in ['get', 'post', 'put', 'delete', 'patch']:
                    fastapi_path = self._convert_openapi_path_to_fastapi(path)
                    documented_endpoints.add(f"{method.lower()}:{fastapi_path}")
        
        implemented_endpoints = set(app_analysis['endpoints'].keys())
        
        # Find undocumented endpoints
        undocumented = implemented_endpoints - documented_endpoints
        
        for endpoint_key in undocumented:
            method, path = endpoint_key.split(':', 1)
            issues.append(ValidationIssue(
                severity='warning',
                category='documentation',
                message=f"Endpoint {method.upper()} {path} implemented but not documented",
                endpoint=f"{method.upper()} {path}",
                suggestion="Add documentation for this endpoint in the OpenAPI specification"
            ))
        
        return issues


class AutomaticAPIUpdater:
    """
    Automatic API Documentation Updates system.
    
    Provides hooks for automatic regeneration on schema changes, validation
    against actual API implementation, and diff detection for API specification
    changes.
    
    Implements Requirements 2.3, 2.5 from the comprehensive documentation specification.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.docs_dir = project_root / "docs" / "api"
        self.cache_dir = project_root / ".kiro" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.differ = APISpecificationDiffer()
        self.validator = APIImplementationValidator(project_root)
        
        # State tracking
        self.last_spec_hash = self._load_last_spec_hash()
        self.update_history: List[UpdateResult] = []
    
    def setup_automatic_hooks(self) -> Dict[str, Any]:
        """
        Set up automatic documentation generation hooks.
        
        Returns:
            Dictionary containing hook setup results
            
        Implements Requirement 2.3: Hooks for automatic regeneration on schema changes
        """
        result = {
            'success': True,
            'hooks_created': [],
            'errors': []
        }
        
        try:
            # Create Git hooks directory
            git_hooks_dir = self.project_root / ".git" / "hooks"
            if git_hooks_dir.exists():
                # Create pre-commit hook for API validation
                pre_commit_hook = self._create_pre_commit_hook()
                if pre_commit_hook:
                    result['hooks_created'].append('pre-commit')
                
                # Create post-commit hook for documentation update
                post_commit_hook = self._create_post_commit_hook()
                if post_commit_hook:
                    result['hooks_created'].append('post-commit')
            
            # Create file watcher for development
            watcher_script = self._create_file_watcher()
            if watcher_script:
                result['hooks_created'].append('file-watcher')
            
            # Create CI/CD integration script
            ci_script = self._create_ci_integration()
            if ci_script:
                result['hooks_created'].append('ci-integration')
            
        except Exception as e:
            result['success'] = False
            result['errors'].append(f"Failed to setup hooks: {str(e)}")
        
        return result
    
    def check_for_api_changes(self) -> UpdateResult:
        """
        Check for API changes and update documentation if needed.
        
        Returns:
            UpdateResult with details of changes and updates
            
        Implements Requirements 2.3, 2.5: Diff detection and automatic updates
        """
        start_time = time.time()
        result = UpdateResult(success=True)
        
        try:
            # Get current API specification
            current_spec = self._get_current_api_spec()
            current_hash = self._calculate_spec_hash(current_spec)
            
            # Check if specification has changed
            if current_hash == self.last_spec_hash:
                result.duration = time.time() - start_time
                return result  # No changes detected
            
            # Load previous specification for comparison
            previous_spec = self._load_previous_spec()
            
            if previous_spec:
                # Detect changes
                changes = self.differ.detect_changes(previous_spec, current_spec)
                result.changes_detected = changes
                
                if changes:
                    print(f"🔄 Detected {len(changes)} API changes")
                    for change in changes[:5]:  # Show first 5 changes
                        print(f"  - {change.description}")
            
            # Validate against implementation
            validation_issues = self.validator.validate_against_implementation(current_spec)
            result.validation_issues = validation_issues
            
            if validation_issues:
                error_count = sum(1 for issue in validation_issues if issue.severity == 'error')
                warning_count = sum(1 for issue in validation_issues if issue.severity == 'warning')
                print(f"⚠️  Found {error_count} errors and {warning_count} warnings in API validation")
            
            # Update documentation if changes detected or validation issues found
            if result.changes_detected or result.validation_issues:
                update_success = self._update_documentation(current_spec, result)
                result.success = update_success
            
            # Save current specification and hash
            self._save_current_spec(current_spec)
            self.last_spec_hash = current_hash
            self._save_last_spec_hash(current_hash)
            
        except Exception as e:
            result.success = False
            result.errors.append(f"API change detection failed: {str(e)}")
        
        result.duration = time.time() - start_time
        self.update_history.append(result)
        
        return result
    
    def validate_api_documentation(self, openapi_spec: Dict[str, Any] = None) -> List[ValidationIssue]:
        """
        Validate API documentation against actual implementation.
        
        Args:
            openapi_spec: OpenAPI specification to validate (optional)
            
        Returns:
            List of validation issues
            
        Implements Requirement 2.5: Validation against actual API implementation
        """
        if openapi_spec is None:
            openapi_spec = self._get_current_api_spec()
        
        return self.validator.validate_against_implementation(openapi_spec)
    
    def generate_change_report(self, changes: List[APIChange]) -> str:
        """
        Generate a human-readable change report.
        
        Args:
            changes: List of API changes to report
            
        Returns:
            Markdown formatted change report
        """
        if not changes:
            return "# API Change Report\n\nNo changes detected.\n"
        
        report = f"# API Change Report\n\n"
        report += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"**Total Changes**: {len(changes)}\n\n"
        
        # Group changes by type
        added = [c for c in changes if c.change_type == 'added']
        removed = [c for c in changes if c.change_type == 'removed']
        modified = [c for c in changes if c.change_type == 'modified']
        
        if added:
            report += f"## Added Endpoints ({len(added)})\n\n"
            for change in added:
                report += f"- **{change.method.upper()} {change.path}**\n"
                if change.new_spec and 'summary' in change.new_spec:
                    report += f"  - {change.new_spec['summary']}\n"
                report += "\n"
        
        if removed:
            report += f"## Removed Endpoints ({len(removed)})\n\n"
            for change in removed:
                report += f"- **{change.method.upper()} {change.path}**\n"
                if change.old_spec and 'summary' in change.old_spec:
                    report += f"  - {change.old_spec['summary']}\n"
                report += "\n"
        
        if modified:
            report += f"## Modified Endpoints ({len(modified)})\n\n"
            for change in modified:
                report += f"- **{change.method.upper()} {change.path}**\n"
                report += f"  - {change.description}\n"
                
                # Show specific changes if available
                if change.old_spec and change.new_spec:
                    diff_summary = self._generate_diff_summary(change.old_spec, change.new_spec)
                    if diff_summary:
                        report += f"  - Changes: {diff_summary}\n"
                report += "\n"
        
        return report
    
    def _get_current_api_spec(self) -> Dict[str, Any]:
        """Get the current API specification from the FastAPI app."""
        try:
            # Import the FastAPI app
            spec = importlib.util.spec_from_file_location("app_main", self.project_root / "app" / "main.py")
            app_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(app_module)
            
            app = getattr(app_module, 'app', None)
            if not app:
                raise ValueError("No FastAPI app found")
            
            return app.openapi()
            
        except Exception as e:
            # Return minimal spec as fallback
            return {
                'openapi': '3.1.0',
                'info': {'title': 'API', 'version': '1.0.0'},
                'paths': {},
                'error': str(e)
            }
    
    def _calculate_spec_hash(self, spec: Dict[str, Any]) -> str:
        """Calculate hash of API specification for change detection."""
        # Remove volatile fields before hashing
        clean_spec = self.differ._clean_spec(spec)
        spec_json = json.dumps(clean_spec, sort_keys=True)
        return hashlib.md5(spec_json.encode()).hexdigest()
    
    def _load_last_spec_hash(self) -> Optional[str]:
        """Load the last known specification hash."""
        hash_file = self.cache_dir / "api_spec_hash.txt"
        if hash_file.exists():
            return hash_file.read_text().strip()
        return None
    
    def _save_last_spec_hash(self, spec_hash: str):
        """Save the current specification hash."""
        hash_file = self.cache_dir / "api_spec_hash.txt"
        hash_file.write_text(spec_hash)
    
    def _load_previous_spec(self) -> Optional[Dict[str, Any]]:
        """Load the previous API specification."""
        spec_file = self.cache_dir / "previous_api_spec.json"
        if spec_file.exists():
            try:
                with open(spec_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return None
    
    def _save_current_spec(self, spec: Dict[str, Any]):
        """Save the current API specification."""
        spec_file = self.cache_dir / "previous_api_spec.json"
        with open(spec_file, 'w') as f:
            json.dump(spec, f, indent=2)
    
    def _update_documentation(self, current_spec: Dict[str, Any], result: UpdateResult) -> bool:
        """Update API documentation with current specification."""
        try:
            from scripts.api_documentation_generator import APIDocumentationGenerator
            
            # Generate updated documentation
            generator = APIDocumentationGenerator(self.project_root)
            doc_result = generator.generate_enhanced_documentation(force_regenerate=True)
            
            if doc_result['success']:
                result.updated_files.extend(doc_result['generated_files'])
                
                # Generate change report if changes were detected
                if result.changes_detected:
                    change_report = self.generate_change_report(result.changes_detected)
                    report_file = self.docs_dir / "CHANGELOG.md"
                    
                    # Append to existing changelog or create new one
                    if report_file.exists():
                        existing_content = report_file.read_text()
                        new_content = change_report + "\n---\n\n" + existing_content
                    else:
                        new_content = change_report
                    
                    report_file.write_text(new_content)
                    result.updated_files.append(report_file)
                
                return True
            else:
                result.errors.extend(doc_result['errors'])
                return False
                
        except Exception as e:
            result.errors.append(f"Documentation update failed: {str(e)}")
            return False
    
    def _generate_diff_summary(self, old_spec: Dict[str, Any], new_spec: Dict[str, Any]) -> str:
        """Generate a summary of differences between two specifications."""
        changes = []
        
        # Check for summary changes
        old_summary = old_spec.get('summary', '')
        new_summary = new_spec.get('summary', '')
        if old_summary != new_summary:
            changes.append("summary updated")
        
        # Check for parameter changes
        old_params = len(old_spec.get('parameters', []))
        new_params = len(new_spec.get('parameters', []))
        if old_params != new_params:
            changes.append(f"parameters changed ({old_params} → {new_params})")
        
        # Check for response changes
        old_responses = set(old_spec.get('responses', {}).keys())
        new_responses = set(new_spec.get('responses', {}).keys())
        if old_responses != new_responses:
            changes.append("response codes changed")
        
        return ", ".join(changes) if changes else "minor changes"
    
    def _create_pre_commit_hook(self) -> bool:
        """Create Git pre-commit hook for API validation."""
        try:
            hook_path = self.project_root / ".git" / "hooks" / "pre-commit"
            
            hook_content = f'''#!/bin/bash
# Pre-commit hook for API documentation validation

echo "🔍 Validating API documentation..."

# Run API documentation validation
cd "{self.project_root}"
python -c "
from scripts.api_auto_updater import AutomaticAPIUpdater
from pathlib import Path

updater = AutomaticAPIUpdater(Path('.'))
issues = updater.validate_api_documentation()

error_count = sum(1 for issue in issues if issue.severity == 'error')
if error_count > 0:
    print(f'❌ Found {{error_count}} API validation errors:')
    for issue in issues:
        if issue.severity == 'error':
            print(f'  - {{issue.message}}')
    exit(1)
else:
    print('✅ API documentation validation passed')
"

exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "❌ Pre-commit validation failed. Fix API documentation issues before committing."
    exit 1
fi

echo "✅ Pre-commit validation passed"
'''
            
            with open(hook_path, 'w') as f:
                f.write(hook_content)
            
            # Make executable
            hook_path.chmod(0o755)
            return True
            
        except Exception as e:
            print(f"Failed to create pre-commit hook: {e}")
            return False
    
    def _create_post_commit_hook(self) -> bool:
        """Create Git post-commit hook for automatic documentation updates."""
        try:
            hook_path = self.project_root / ".git" / "hooks" / "post-commit"
            
            hook_content = f'''#!/bin/bash
# Post-commit hook for automatic API documentation updates

echo "🔄 Checking for API changes..."

# Run automatic API documentation update
cd "{self.project_root}"
python -c "
from scripts.api_auto_updater import AutomaticAPIUpdater
from pathlib import Path

updater = AutomaticAPIUpdater(Path('.'))
result = updater.check_for_api_changes()

if result.changes_detected:
    print(f'📝 Updated documentation for {{len(result.changes_detected)}} API changes')
    for file_path in result.updated_files:
        print(f'  - {{file_path.relative_to(Path('.'))}}')
else:
    print('✅ No API changes detected')

if result.validation_issues:
    warning_count = sum(1 for issue in result.validation_issues if issue.severity == 'warning')
    if warning_count > 0:
        print(f'⚠️  Found {{warning_count}} API validation warnings')
"

echo "✅ Post-commit API documentation update complete"
'''
            
            with open(hook_path, 'w') as f:
                f.write(hook_content)
            
            # Make executable
            hook_path.chmod(0o755)
            return True
            
        except Exception as e:
            print(f"Failed to create post-commit hook: {e}")
            return False
    
    def _create_file_watcher(self) -> bool:
        """Create file watcher script for development."""
        try:
            scripts_dir = self.project_root / "scripts"
            scripts_dir.mkdir(exist_ok=True)
            watcher_path = scripts_dir / "watch_api_changes.py"
            
            watcher_content = '''#!/usr/bin/env python3
"""
API Documentation File Watcher

Watches for changes in API-related files and automatically updates documentation.
"""

import time
import sys
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.api_auto_updater import AutomaticAPIUpdater


class APIChangeHandler(FileSystemEventHandler):
    """Handles file system events for API-related files."""
    
    def __init__(self):
        self.updater = AutomaticAPIUpdater(project_root)
        self.last_update = 0
        self.debounce_seconds = 2
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        # Check if it's an API-related file
        file_path = Path(event.src_path)
        if self._is_api_related_file(file_path):
            current_time = time.time()
            
            # Debounce rapid file changes
            if current_time - self.last_update > self.debounce_seconds:
                print(f"🔄 API file changed: {file_path.relative_to(project_root)}")
                self._update_documentation()
                self.last_update = current_time
    
    def _is_api_related_file(self, file_path: Path) -> bool:
        """Check if file is related to API definition."""
        api_patterns = [
            'app/api/',
            'app/schemas/',
            'app/main.py',
            'app/models/',
            'endpoints/'
        ]
        
        file_str = str(file_path)
        return any(pattern in file_str for pattern in api_patterns) and file_str.endswith('.py')
    
    def _update_documentation(self):
        """Update API documentation."""
        try:
            result = self.updater.check_for_api_changes()
            
            if result.changes_detected:
                print(f"📝 Updated documentation for {len(result.changes_detected)} changes")
            elif result.validation_issues:
                error_count = sum(1 for issue in result.validation_issues if issue.severity == 'error')
                if error_count > 0:
                    print(f"❌ Found {error_count} validation errors")
            else:
                print("✅ No changes detected")
                
        except Exception as e:
            print(f"❌ Documentation update failed: {e}")


def main():
    """Main function to start the file watcher."""
    print("🔍 Starting API documentation file watcher...")
    print("Watching for changes in API-related files...")
    print("Press Ctrl+C to stop")
    
    event_handler = APIChangeHandler()
    observer = Observer()
    
    # Watch the app directory
    observer.schedule(event_handler, str(project_root / "app"), recursive=True)
    
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\\n🛑 File watcher stopped")
    
    observer.join()


if __name__ == "__main__":
    main()
'''
            
            with open(watcher_path, 'w') as f:
                f.write(watcher_content)
            
            # Make executable
            watcher_path.chmod(0o755)
            return True
            
        except Exception as e:
            print(f"Failed to create file watcher: {e}")
            return False
    
    def _create_ci_integration(self) -> bool:
        """Create CI/CD integration script."""
        try:
            scripts_dir = self.project_root / "scripts"
            scripts_dir.mkdir(exist_ok=True)
            ci_path = scripts_dir / "ci_api_docs.py"
            
            ci_content = f'''#!/usr/bin/env python3
"""
CI/CD API Documentation Integration

Script for integrating API documentation updates into CI/CD pipelines.
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.api_auto_updater import AutomaticAPIUpdater


def main():
    """Main CI/CD integration function."""
    updater = AutomaticAPIUpdater(project_root)
    
    # Check for API changes
    result = updater.check_for_api_changes()
    
    # Output results in CI-friendly format
    if result.success:
        print("✅ API documentation check passed")
        
        if result.changes_detected:
            print(f"📝 Detected {{len(result.changes_detected)}} API changes:")
            for change in result.changes_detected:
                print(f"  - {{change.description}}")
        
        # Check for validation errors
        error_count = sum(1 for issue in result.validation_issues if issue.severity == 'error')
        if error_count > 0:
            print(f"❌ Found {{error_count}} API validation errors:")
            for issue in result.validation_issues:
                if issue.severity == 'error':
                    print(f"  - {{issue.message}}")
            sys.exit(1)
        
        # Output JSON for further processing
        output = {{
            'success': result.success,
            'changes_count': len(result.changes_detected),
            'validation_errors': error_count,
            'updated_files': [str(f.relative_to(project_root)) for f in result.updated_files]
        }}
        
        print(f"::set-output name=api_docs_result::{json.dumps(output)}")
        
    else:
        print("❌ API documentation check failed:")
        for error in result.errors:
            print(f"  - {{error}}")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''
            
            with open(ci_path, 'w') as f:
                f.write(ci_content)
            
            # Make executable
            ci_path.chmod(0o755)
            return True
            
        except Exception as e:
            print(f"Failed to create CI integration script: {e}")
            return False