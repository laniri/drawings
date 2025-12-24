#!/usr/bin/env python3
"""
Documentation Generation Script

This script automatically generates and updates documentation from code,
ensuring consistency between implementation and documentation.
"""

import os
import sys
import json
import subprocess
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
import ast
import inspect
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import C4 templates
from scripts.c4_templates import C4DiagramTemplates, C4DiagramGenerator

# Import workflow generator
from scripts.workflow_documentation_generator import WorkflowGenerator

# Import validation engine
from scripts.validation_engine import ValidationEngine as ComprehensiveValidationEngine

# Import search and navigation engines
from scripts.search_engine import DocumentationSearchEngine
from scripts.navigation_engine import NavigationEngine


class DocumentationType(Enum):
    """Types of documentation that can be generated."""
    API = "api"
    SERVICES = "services"
    ALGORITHMS = "algorithms"
    DATABASE = "database"
    FRONTEND = "frontend"
    DEPLOYMENT = "deployment"
    ARCHITECTURE = "architecture"
    WORKFLOWS = "workflows"
    INTERFACES = "interfaces"


@dataclass
class FileChange:
    """Represents a change to a source file."""
    path: Path
    change_type: str  # 'created', 'modified', 'deleted'
    timestamp: datetime
    content_hash: Optional[str] = None
    dependencies: Set[str] = field(default_factory=set)


@dataclass
class GenerationResult:
    """Result of a documentation generation operation."""
    success: bool
    generated_files: List[Path] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration: float = 0.0
    changes_detected: List[FileChange] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of documentation validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validated_files: List[Path] = field(default_factory=list)


class ChangeDetector:
    """Detects changes in source files for incremental updates."""
    
    def __init__(self, cache_file: Path):
        self.cache_file = cache_file
        self.file_cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load file cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {}
    
    def _save_cache(self):
        """Save file cache to disk."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.file_cache, f, indent=2, default=str)
    
    def _get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get file information for change detection."""
        if not file_path.exists():
            return {}
        
        stat = file_path.stat()
        content_hash = hashlib.md5(file_path.read_bytes()).hexdigest()
        
        return {
            'mtime': stat.st_mtime,
            'size': stat.st_size,
            'hash': content_hash
        }
    
    def detect_changes(self, source_paths: List[Path]) -> List[FileChange]:
        """Detect changes in source files."""
        changes = []
        current_files = set()
        
        for source_path in source_paths:
            if source_path.is_file():
                current_files.add(str(source_path))
                self._check_file_changes(source_path, changes)
            elif source_path.is_dir():
                for file_path in source_path.rglob("*.py"):
                    current_files.add(str(file_path))
                    self._check_file_changes(file_path, changes)
        
        # Check for deleted files
        cached_files = set(self.file_cache.keys())
        deleted_files = cached_files - current_files
        
        for deleted_file in deleted_files:
            changes.append(FileChange(
                path=Path(deleted_file),
                change_type='deleted',
                timestamp=datetime.now()
            ))
            del self.file_cache[deleted_file]
        
        self._save_cache()
        return changes
    
    def _check_file_changes(self, file_path: Path, changes: List[FileChange]):
        """Check if a single file has changed."""
        file_key = str(file_path)
        current_info = self._get_file_info(file_path)
        cached_info = self.file_cache.get(file_key, {})
        
        if not cached_info:
            # New file
            changes.append(FileChange(
                path=file_path,
                change_type='created',
                timestamp=datetime.fromtimestamp(current_info.get('mtime', time.time())),
                content_hash=current_info.get('hash')
            ))
        elif (current_info.get('hash') != cached_info.get('hash') or
              current_info.get('mtime') != cached_info.get('mtime')):
            # Modified file
            changes.append(FileChange(
                path=file_path,
                change_type='modified',
                timestamp=datetime.fromtimestamp(current_info.get('mtime', time.time())),
                content_hash=current_info.get('hash')
            ))
        
        # Update cache
        self.file_cache[file_key] = current_info


@dataclass
class CacheEntry:
    """Represents a cached documentation component."""
    content_hash: str
    generated_files: List[Path]
    timestamp: datetime
    dependencies: Set[str] = field(default_factory=set)


class DocumentationCache:
    """Manages caching of generated documentation components."""
    
    def __init__(self, cache_file: Path):
        self.cache_file = cache_file
        self.cache: Dict[str, CacheEntry] = self._load_cache()
    
    def _load_cache(self) -> Dict[str, CacheEntry]:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    cache = {}
                    for key, entry_data in data.items():
                        cache[key] = CacheEntry(
                            content_hash=entry_data['content_hash'],
                            generated_files=[Path(p) for p in entry_data['generated_files']],
                            timestamp=datetime.fromisoformat(entry_data['timestamp']),
                            dependencies=set(entry_data.get('dependencies', []))
                        )
                    return cache
            except (json.JSONDecodeError, IOError, KeyError):
                pass
        return {}
    
    def _save_cache(self):
        """Save cache to disk."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for key, entry in self.cache.items():
            data[key] = {
                'content_hash': entry.content_hash,
                'generated_files': [str(p) for p in entry.generated_files],
                'timestamp': entry.timestamp.isoformat(),
                'dependencies': list(entry.dependencies)
            }
        
        with open(self.cache_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_entry(self, component: str) -> Optional[CacheEntry]:
        """Get cache entry for a component."""
        return self.cache.get(component)
    
    def set_entry(self, component: str, entry: CacheEntry):
        """Set cache entry for a component."""
        self.cache[component] = entry
        self._save_cache()
    
    def invalidate_component(self, component: str):
        """Invalidate cache for a specific component."""
        if component in self.cache:
            del self.cache[component]
            self._save_cache()
    
    def is_valid(self, component: str, source_hash: str) -> bool:
        """Check if cached component is still valid."""
        entry = self.cache.get(component)
        if not entry:
            return False
        
        # Check if source hash matches
        if entry.content_hash != source_hash:
            return False
        
        # Check if generated files still exist
        for file_path in entry.generated_files:
            if not file_path.exists():
                return False
        
        return True


class DependencyGraph:
    """Manages dependency relationships between documentation components."""
    
    def __init__(self):
        self.dependencies: Dict[str, Set[str]] = {}
        self.reverse_dependencies: Dict[str, Set[str]] = {}
        self.file_patterns: Dict[str, Set[str]] = {}
    
    def add_dependency(self, dependent: str, dependency: str):
        """Add a dependency relationship."""
        if dependent not in self.dependencies:
            self.dependencies[dependent] = set()
        self.dependencies[dependent].add(dependency)
        
        if dependency not in self.reverse_dependencies:
            self.reverse_dependencies[dependency] = set()
        self.reverse_dependencies[dependency].add(dependent)
    
    def add_file_pattern(self, component: str, pattern: str):
        """Add a file pattern that affects a component."""
        if component not in self.file_patterns:
            self.file_patterns[component] = set()
        self.file_patterns[component].add(pattern)
    
    def get_affected_components(self, changed_files: List[FileChange]) -> Set[DocumentationType]:
        """Get documentation components affected by file changes."""
        affected = set()
        
        for change in changed_files:
            file_path = str(change.path)
            
            # Check file patterns
            for component, patterns in self.file_patterns.items():
                for pattern in patterns:
                    if pattern in file_path:
                        try:
                            affected.add(DocumentationType(component))
                        except ValueError:
                            pass
            
            # Direct mapping of file patterns to documentation types
            if 'api' in file_path or 'endpoints' in file_path:
                affected.add(DocumentationType.API)
            if 'services' in file_path:
                affected.add(DocumentationType.SERVICES)
                affected.add(DocumentationType.ALGORITHMS)
            if 'models' in file_path or 'database' in file_path:
                affected.add(DocumentationType.DATABASE)
            if 'frontend' in file_path or file_path.endswith('.tsx') or file_path.endswith('.ts'):
                affected.add(DocumentationType.FRONTEND)
        
        # Add dependent components
        initial_affected = affected.copy()
        for component in initial_affected:
            affected.update(self._get_transitive_dependents(component.value))
        
        return affected
    
    def _get_transitive_dependents(self, component: str) -> Set[DocumentationType]:
        """Get all components that transitively depend on the given component."""
        dependents = set()
        to_process = {component}
        processed = set()
        
        while to_process:
            current = to_process.pop()
            if current in processed:
                continue
            processed.add(current)
            
            direct_dependents = self.reverse_dependencies.get(current, set())
            for dependent in direct_dependents:
                try:
                    dependents.add(DocumentationType(dependent))
                    to_process.add(dependent)
                except ValueError:
                    pass
        
        return dependents
    
    def get_generation_order(self, components: Set[DocumentationType]) -> List[DocumentationType]:
        """Get the order in which components should be generated based on dependencies."""
        # Topological sort of components
        in_degree = {comp: 0 for comp in components}
        
        # Calculate in-degrees
        for comp in components:
            deps = self.dependencies.get(comp.value, set())
            for dep in deps:
                try:
                    dep_type = DocumentationType(dep)
                    if dep_type in components:
                        in_degree[comp] += 1
                except ValueError:
                    pass
        
        # Generate order using Kahn's algorithm
        queue = [comp for comp in components if in_degree[comp] == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # Update in-degrees of dependents
            dependents = self.reverse_dependencies.get(current.value, set())
            for dependent in dependents:
                try:
                    dep_type = DocumentationType(dependent)
                    if dep_type in components:
                        in_degree[dep_type] -= 1
                        if in_degree[dep_type] == 0:
                            queue.append(dep_type)
                except ValueError:
                    pass
        
        return result


class DependencyManager(DependencyGraph):
    """Legacy alias for backward compatibility."""
    pass


class ArchitectureGenerator:
    """
    C4 Model architecture documentation generator.
    
    Generates comprehensive C4 model documentation including system context (Level 1),
    container diagrams (Level 2), component diagrams (Level 3), and code diagrams (Level 4)
    from codebase analysis.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.app_dir = project_root / "app"
        self.frontend_dir = project_root / "frontend"
        self.docs_dir = project_root / "docs"
        
        # Initialize C4 diagram generator
        self.c4_generator = C4DiagramGenerator()
        self.templates = C4DiagramTemplates()
        
    def generate_system_context(self) -> Dict[str, Any]:
        """
        Generate C4 Level 1 system context diagram.
        
        Analyzes the project structure to identify external systems, users,
        and system boundaries for the context diagram.
        
        Returns:
            Dictionary containing system context information
        """
        # Analyze project configuration to identify external systems
        external_systems = []
        users = []
        
        # Standard external systems for ML applications
        external_systems.extend([
            "File Storage System",
            "ML Model Repository"
        ])
        
        # Check for cloud services
        if (self.project_root / "requirements.txt").exists():
            req_content = (self.project_root / "requirements.txt").read_text().lower()
            if 'boto3' in req_content or 'aws' in req_content:
                external_systems.append("AWS Services")
        
        # Check for Docker deployment
        if (self.project_root / "docker-compose.yml").exists():
            external_systems.append("Container Registry")
        
        # Standard users for drawing analysis system
        users.extend([
            "Researcher",
            "Educational Professional", 
            "Healthcare Provider"
        ])
        
        # Determine system name from project structure
        system_name = "Children's Drawing Anomaly Detection System"
        if (self.project_root / "package.json").exists():
            try:
                import json
                with open(self.project_root / "package.json") as f:
                    package_data = json.load(f)
                    system_name = package_data.get("name", system_name)
            except (json.JSONDecodeError, IOError):
                pass
        
        return {
            'level': 1,
            'type': 'system_context',
            'external_systems': external_systems,
            'users': users,
            'system_name': system_name,
            'description': 'ML-powered system for analyzing children\'s drawings to detect developmental anomalies'
        }
    
    def generate_container_diagram(self) -> Dict[str, Any]:
        """
        Generate C4 Level 2 container diagram.
        
        Analyzes the project structure to identify high-level containers
        and their technology choices.
        
        Returns:
            Dictionary containing container information
        """
        containers = []
        
        # Detect backend container
        if self.app_dir.exists():
            # Analyze backend technology
            technology = "Python"
            framework = "FastAPI"
            
            if (self.app_dir / "main.py").exists():
                try:
                    main_content = (self.app_dir / "main.py").read_text()
                    if 'fastapi' in main_content.lower():
                        framework = "FastAPI"
                    elif 'flask' in main_content.lower():
                        framework = "Flask"
                    elif 'django' in main_content.lower():
                        framework = "Django"
                except IOError:
                    pass
            
            containers.append({
                'name': 'Backend API',
                'technology': f'{framework}/{technology}',
                'description': 'REST API server providing ML analysis and data management',
                'responsibilities': [
                    'Drawing upload and processing',
                    'ML model inference',
                    'Anomaly detection scoring',
                    'Data persistence'
                ]
            })
        
        # Detect frontend container
        if self.frontend_dir.exists():
            technology = "React"
            
            # Check package.json for frontend technology
            package_json = self.frontend_dir / "package.json"
            if package_json.exists():
                try:
                    import json
                    with open(package_json) as f:
                        package_data = json.load(f)
                        deps = package_data.get("dependencies", {})
                        
                        if "react" in deps:
                            technology = "React"
                        elif "vue" in deps:
                            technology = "Vue.js"
                        elif "angular" in deps:
                            technology = "Angular"
                        
                        # Check for TypeScript
                        if "typescript" in deps or "@types/react" in deps:
                            technology += "/TypeScript"
                        else:
                            technology += "/JavaScript"
                            
                except (json.JSONDecodeError, IOError):
                    pass
            
            containers.append({
                'name': 'Web Application',
                'technology': technology,
                'description': 'Single-page application providing user interface',
                'responsibilities': [
                    'Drawing upload interface',
                    'Analysis results visualization',
                    'Configuration management',
                    'Dashboard and reporting'
                ]
            })
        
        # Detect database container
        database_detected = False
        if (self.project_root / "requirements.txt").exists():
            try:
                req_content = (self.project_root / "requirements.txt").read_text().lower()
                
                if 'sqlalchemy' in req_content:
                    db_tech = "SQLite/SQLAlchemy"
                    if 'postgresql' in req_content or 'psycopg' in req_content:
                        db_tech = "PostgreSQL/SQLAlchemy"
                    elif 'mysql' in req_content:
                        db_tech = "MySQL/SQLAlchemy"
                    
                    containers.append({
                        'name': 'Database',
                        'technology': db_tech,
                        'description': 'Relational database for metadata and results storage',
                        'responsibilities': [
                            'Drawing metadata storage',
                            'Analysis results persistence',
                            'Configuration data',
                            'User session management'
                        ]
                    })
                    database_detected = True
                    
            except IOError:
                pass
        
        # Check for file storage container
        if (self.project_root / "uploads").exists() or (self.project_root / "static").exists():
            containers.append({
                'name': 'File Storage',
                'technology': 'Local Filesystem',
                'description': 'File storage for drawings and generated content',
                'responsibilities': [
                    'Drawing file storage',
                    'ML model storage',
                    'Generated visualizations',
                    'Backup and export files'
                ]
            })
        
        # Check for ML model container
        if (self.project_root / "static" / "models").exists():
            containers.append({
                'name': 'ML Model Store',
                'technology': 'PyTorch/Pickle',
                'description': 'Storage and management of trained ML models',
                'responsibilities': [
                    'Autoencoder model storage',
                    'ViT feature extractor',
                    'Model versioning',
                    'Training artifacts'
                ]
            })
        
        return {
            'level': 2,
            'type': 'container',
            'containers': containers
        }
    
    def generate_component_diagram(self, service: str = None) -> Dict[str, Any]:
        """
        Generate C4 Level 3 component diagram.
        
        Analyzes the service layer and other components to show
        internal structure and relationships.
        
        Args:
            service: Optional specific service to focus on
            
        Returns:
            Dictionary containing component information
        """
        components = []
        
        # Analyze service layer components
        if self.app_dir.exists():
            services_dir = self.app_dir / "services"
            if services_dir.exists():
                for service_file in services_dir.glob("*.py"):
                    if service_file.name != "__init__.py":
                        component_info = self._analyze_service_component(service_file)
                        if component_info:
                            components.append(component_info)
            
            # Analyze API layer
            api_dir = self.app_dir / "api"
            if api_dir.exists():
                components.append({
                    'name': 'API Router',
                    'type': 'api',
                    'technology': 'FastAPI',
                    'description': 'HTTP request routing and validation',
                    'responsibilities': [
                        'Request/response handling',
                        'Input validation',
                        'Authentication',
                        'Error handling'
                    ],
                    'interfaces': ['REST API', 'OpenAPI Schema']
                })
                
                # Analyze specific endpoints
                endpoints_dir = api_dir / "api_v1" / "endpoints"
                if endpoints_dir.exists():
                    endpoint_files = list(endpoints_dir.glob("*.py"))
                    if endpoint_files:
                        components.append({
                            'name': 'API Endpoints',
                            'type': 'endpoint_collection',
                            'technology': 'FastAPI',
                            'description': f'Collection of {len(endpoint_files)} API endpoint modules',
                            'responsibilities': [
                                'Business logic orchestration',
                                'Service layer integration',
                                'Response formatting'
                            ],
                            'endpoints': [f.stem for f in endpoint_files if f.name != "__init__.py"]
                        })
            
            # Analyze data models
            models_dir = self.app_dir / "models"
            if models_dir.exists():
                components.append({
                    'name': 'Data Models',
                    'type': 'model',
                    'technology': 'SQLAlchemy',
                    'description': 'Database schema and ORM models',
                    'responsibilities': [
                        'Data structure definition',
                        'Database relationships',
                        'Query interface',
                        'Migration support'
                    ],
                    'interfaces': ['SQLAlchemy ORM', 'Database Schema']
                })
            
            # Analyze schemas (Pydantic models)
            schemas_dir = self.app_dir / "schemas"
            if schemas_dir.exists():
                schema_files = list(schemas_dir.glob("*.py"))
                if schema_files:
                    components.append({
                        'name': 'Data Schemas',
                        'type': 'schema',
                        'technology': 'Pydantic',
                        'description': 'Request/response validation schemas',
                        'responsibilities': [
                            'Input validation',
                            'Serialization/deserialization',
                            'Type checking',
                            'API documentation'
                        ],
                        'schemas': [f.stem for f in schema_files if f.name != "__init__.py"]
                    })
            
            # Analyze core configuration
            core_dir = self.app_dir / "core"
            if core_dir.exists():
                components.append({
                    'name': 'Core Configuration',
                    'type': 'configuration',
                    'technology': 'Pydantic Settings',
                    'description': 'Application configuration and settings management',
                    'responsibilities': [
                        'Environment configuration',
                        'Database connection',
                        'Security settings',
                        'Feature flags'
                    ],
                    'interfaces': ['Settings API', 'Environment Variables']
                })
        
        return {
            'level': 3,
            'type': 'component',
            'components': components,
            'relationships': self._analyze_component_relationships(components)
        }
    
    def _analyze_service_component(self, service_file: Path) -> Dict[str, Any]:
        """Analyze a single service file to extract component information."""
        try:
            content = service_file.read_text()
            tree = ast.parse(content)
            
            # Extract module docstring
            module_doc = ast.get_docstring(tree) or ""
            
            # Extract classes and their purposes
            classes = []
            methods = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_doc = ast.get_docstring(node) or ""
                    class_methods = []
                    
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and not item.name.startswith('_'):
                            method_doc = ast.get_docstring(item) or ""
                            class_methods.append({
                                'name': item.name,
                                'description': method_doc.split('\n')[0] if method_doc else ""
                            })
                    
                    classes.append({
                        'name': node.name,
                        'description': class_doc.split('\n')[0] if class_doc else "",
                        'methods': class_methods
                    })
                    methods.extend(class_methods)
            
            # Determine component type and responsibilities based on filename and content
            component_name = service_file.stem.replace('_', ' ').title()
            component_type = 'service'
            
            # Categorize service types
            if 'manager' in service_file.stem:
                component_type = 'manager'
            elif 'engine' in service_file.stem:
                component_type = 'engine'
            elif 'service' in service_file.stem:
                component_type = 'service'
            elif 'pipeline' in service_file.stem:
                component_type = 'pipeline'
            
            # Extract responsibilities from method names and docstrings
            responsibilities = []
            for method in methods:
                if method['description']:
                    responsibilities.append(method['description'])
                else:
                    # Infer from method name
                    method_name = method['name'].replace('_', ' ')
                    responsibilities.append(f"{method_name.title()} functionality")
            
            # Limit to most important responsibilities
            responsibilities = responsibilities[:4]
            
            return {
                'name': component_name,
                'type': component_type,
                'technology': 'Python',
                'description': module_doc.split('\n')[0] if module_doc else f"{component_name} implementation",
                'file': str(service_file.relative_to(self.project_root)),
                'classes': classes,
                'responsibilities': responsibilities,
                'interfaces': [f"{component_name} API"]
            }
            
        except (SyntaxError, UnicodeDecodeError, IOError):
            return None
    
    def _analyze_component_relationships(self, components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze relationships between components."""
        relationships = []
        
        # Standard relationships for layered architecture
        api_components = [c for c in components if c.get('type') == 'api']
        service_components = [c for c in components if c.get('type') in ['service', 'manager', 'engine']]
        model_components = [c for c in components if c.get('type') == 'model']
        
        # API layer depends on service layer
        for api_comp in api_components:
            for service_comp in service_components:
                relationships.append({
                    'from': api_comp['name'],
                    'to': service_comp['name'],
                    'type': 'uses',
                    'description': 'Orchestrates business logic'
                })
        
        # Service layer depends on model layer
        for service_comp in service_components:
            for model_comp in model_components:
                relationships.append({
                    'from': service_comp['name'],
                    'to': model_comp['name'],
                    'type': 'uses',
                    'description': 'Persists and retrieves data'
                })
        
        return relationships
    
    def generate_code_diagram(self, module: str = None) -> Dict[str, Any]:
        """
        Generate C4 Level 4 code diagram.
        
        Analyzes Python files to extract detailed class structures,
        interfaces, and dependencies for code-level documentation.
        
        Args:
            module: Optional specific module to focus on
            
        Returns:
            Dictionary containing code structure information
        """
        classes = []
        relationships = []
        interfaces = []
        
        # Analyze Python files for class structures
        if self.app_dir.exists():
            target_dirs = [self.app_dir]
            if module:
                # Focus on specific module if provided
                module_path = self.app_dir / module
                if module_path.exists():
                    target_dirs = [module_path]
            
            for target_dir in target_dirs:
                for py_file in target_dir.rglob("*.py"):
                    if py_file.name != "__init__.py":
                        file_analysis = self._analyze_python_file(py_file)
                        classes.extend(file_analysis['classes'])
                        relationships.extend(file_analysis['relationships'])
                        interfaces.extend(file_analysis['interfaces'])
        
        return {
            'level': 4,
            'type': 'code',
            'classes': classes,
            'relationships': relationships,
            'interfaces': interfaces,
            'module_focus': module
        }
    
    def _analyze_python_file(self, py_file: Path) -> Dict[str, Any]:
        """Analyze a Python file for classes, methods, and relationships."""
        analysis = {
            'classes': [],
            'relationships': [],
            'interfaces': []
        }
        
        try:
            content = py_file.read_text()
            tree = ast.parse(content)
            
            # Extract imports for dependency analysis
            imports = self._extract_imports(tree)
            
            # Extract classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self._analyze_class(node, py_file, imports)
                    analysis['classes'].append(class_info)
                    
                    # Extract relationships from this class
                    class_relationships = self._extract_class_relationships(node, class_info['name'], imports)
                    analysis['relationships'].extend(class_relationships)
                    
                    # Extract interfaces (abstract methods, protocols)
                    class_interfaces = self._extract_class_interfaces(node, class_info['name'])
                    analysis['interfaces'].extend(class_interfaces)
        
        except (SyntaxError, UnicodeDecodeError, IOError):
            # Skip files that can't be parsed
            pass
        
        return analysis
    
    def _extract_imports(self, tree: ast.AST) -> Dict[str, str]:
        """Extract import statements for dependency analysis."""
        imports = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        full_name = f"{node.module}.{alias.name}"
                        imports[alias.asname or alias.name] = full_name
        
        return imports
    
    def _analyze_class(self, class_node: ast.ClassDef, file_path: Path, imports: Dict[str, str]) -> Dict[str, Any]:
        """Analyze a class definition for detailed information."""
        class_info = {
            'name': class_node.name,
            'file': str(file_path.relative_to(self.project_root)),
            'line_number': class_node.lineno,
            'docstring': ast.get_docstring(class_node) or "",
            'methods': [],
            'attributes': [],
            'base_classes': [],
            'decorators': []
        }
        
        # Extract base classes
        for base in class_node.bases:
            if isinstance(base, ast.Name):
                class_info['base_classes'].append(base.id)
            elif isinstance(base, ast.Attribute):
                class_info['base_classes'].append(f"{base.value.id}.{base.attr}")
        
        # Extract decorators
        for decorator in class_node.decorator_list:
            if isinstance(decorator, ast.Name):
                class_info['decorators'].append(decorator.id)
        
        # Extract methods
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._analyze_method(item)
                class_info['methods'].append(method_info)
            elif isinstance(item, ast.Assign):
                # Extract class attributes
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        class_info['attributes'].append({
                            'name': target.id,
                            'line_number': item.lineno
                        })
        
        return class_info
    
    def _analyze_method(self, method_node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze a method definition."""
        method_info = {
            'name': method_node.name,
            'line_number': method_node.lineno,
            'docstring': ast.get_docstring(method_node) or "",
            'parameters': [],
            'return_annotation': None,
            'decorators': [],
            'is_property': False,
            'is_static': False,
            'is_class_method': False
        }
        
        # Extract parameters
        for arg in method_node.args.args:
            param_info = {'name': arg.arg}
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    param_info['type'] = arg.annotation.id
                elif isinstance(arg.annotation, ast.Constant):
                    param_info['type'] = str(arg.annotation.value)
            method_info['parameters'].append(param_info)
        
        # Extract return annotation
        if method_node.returns:
            if isinstance(method_node.returns, ast.Name):
                method_info['return_annotation'] = method_node.returns.id
            elif isinstance(method_node.returns, ast.Constant):
                method_info['return_annotation'] = str(method_node.returns.value)
        
        # Extract decorators
        for decorator in method_node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorator_name = decorator.id
                method_info['decorators'].append(decorator_name)
                
                # Check for special method types
                if decorator_name == 'property':
                    method_info['is_property'] = True
                elif decorator_name == 'staticmethod':
                    method_info['is_static'] = True
                elif decorator_name == 'classmethod':
                    method_info['is_class_method'] = True
        
        return method_info
    
    def _extract_class_relationships(self, class_node: ast.ClassDef, class_name: str, imports: Dict[str, str]) -> List[Dict[str, Any]]:
        """Extract relationships from class definition."""
        relationships = []
        
        # Inheritance relationships
        for base in class_node.bases:
            base_name = None
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = f"{base.value.id}.{base.attr}"
            
            if base_name:
                relationships.append({
                    'from': class_name,
                    'to': base_name,
                    'type': 'inherits',
                    'description': f'{class_name} inherits from {base_name}'
                })
        
        # Composition/dependency relationships from __init__ method
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                for stmt in ast.walk(item):
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                                if target.value.id == 'self':
                                    # Look for service/dependency injection patterns
                                    if isinstance(stmt.value, ast.Call):
                                        if isinstance(stmt.value.func, ast.Name):
                                            dependency_class = stmt.value.func.id
                                            relationships.append({
                                                'from': class_name,
                                                'to': dependency_class,
                                                'type': 'depends_on',
                                                'description': f'{class_name} depends on {dependency_class}',
                                                'attribute': target.attr
                                            })
        
        return relationships
    
    def _extract_class_interfaces(self, class_node: ast.ClassDef, class_name: str) -> List[Dict[str, Any]]:
        """Extract interface definitions from class."""
        interfaces = []
        
        # Check if class defines an interface (abstract methods, protocols)
        has_abstract_methods = False
        public_methods = []
        
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef):
                # Check for abstract methods
                for decorator in item.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == 'abstractmethod':
                        has_abstract_methods = True
                
                # Collect public methods (interface)
                if not item.name.startswith('_'):
                    method_signature = self._get_method_signature(item)
                    public_methods.append(method_signature)
        
        # If class has abstract methods or follows interface patterns, create interface
        if has_abstract_methods or len(public_methods) > 2:  # More than just __init__ and one method
            interfaces.append({
                'name': f'{class_name}Interface',
                'class': class_name,
                'type': 'abstract' if has_abstract_methods else 'protocol',
                'methods': public_methods,
                'description': f'Interface defined by {class_name}'
            })
        
        return interfaces
    
    def _get_method_signature(self, method_node: ast.FunctionDef) -> str:
        """Generate method signature string."""
        params = []
        for arg in method_node.args.args:
            param = arg.arg
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    param += f": {arg.annotation.id}"
            params.append(param)
        
        signature = f"{method_node.name}({', '.join(params)})"
        
        if method_node.returns:
            if isinstance(method_node.returns, ast.Name):
                signature += f" -> {method_node.returns.id}"
        
        return signature


class APIDocumentationGenerator:
    """
    Enhanced API documentation generator with comprehensive features.
    
    Generates OpenAPI 3.0+ compliant specifications, interactive Swagger UI,
    comprehensive request/response examples, authentication requirements,
    and automatic schema validation and updates.
    """
    
    def __init__(self, project_root: Path, docs_dir: Path):
        self.project_root = project_root
        self.docs_dir = docs_dir
        self.api_dir = docs_dir / "api"
        self.api_dir.mkdir(exist_ok=True)
        
    def generate_comprehensive_docs(self):
        """Generate comprehensive API documentation with all enhancements."""
        try:
            # Import FastAPI app to get OpenAPI schema
            from app.main import app
            
            # Extract and validate OpenAPI schema
            openapi_schema = self._extract_and_validate_schema(app)
            
            # Save enhanced OpenAPI schema
            self._save_openapi_schema(openapi_schema)
            
            # Generate enhanced endpoint documentation
            self._generate_enhanced_endpoint_docs(openapi_schema)
            
            # Generate interactive Swagger UI
            self._generate_swagger_ui(openapi_schema)
            
            # Generate API overview and index
            self._generate_api_overview(openapi_schema)
            
            # Generate authentication documentation
            self._generate_auth_documentation(openapi_schema)
            
            # Generate error specifications
            self._generate_error_specifications(openapi_schema)
            
        except Exception as e:
            raise Exception(f"API documentation generation failed: {str(e)}")
    
    def _extract_and_validate_schema(self, app) -> Dict[str, Any]:
        """Extract OpenAPI schema and validate compliance."""
        openapi_schema = app.openapi()
        
        # Validate OpenAPI 3.0+ compliance
        required_fields = ['openapi', 'info', 'paths']
        for field in required_fields:
            if field not in openapi_schema:
                raise ValueError(f"Invalid OpenAPI schema: missing required field '{field}'")
        
        # Ensure OpenAPI version is 3.0+
        version = openapi_schema.get('openapi', '3.0.0')
        if not version.startswith('3.'):
            openapi_schema['openapi'] = '3.1.0'
        
        # Enhance schema with additional metadata
        openapi_schema = self._enhance_schema_metadata(openapi_schema)
        
        return openapi_schema
    
    def _enhance_schema_metadata(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance OpenAPI schema with additional metadata and examples."""
        # Add server information if missing
        if 'servers' not in schema:
            schema['servers'] = [
                {
                    'url': 'http://localhost:8000',
                    'description': 'Development server'
                },
                {
                    'url': '/api/v1',
                    'description': 'API v1 base path'
                }
            ]
        
        # Enhance info section
        info = schema.get('info', {})
        if 'contact' not in info:
            info['contact'] = {
                'name': 'API Support',
                'email': 'support@example.com'
            }
        
        if 'license' not in info:
            info['license'] = {
                'name': 'MIT',
                'url': 'https://opensource.org/licenses/MIT'
            }
        
        schema['info'] = info
        
        # Add comprehensive examples to paths
        paths = schema.get('paths', {})
        for path, methods in paths.items():
            for method, spec in methods.items():
                if isinstance(spec, dict):
                    spec = self._enhance_endpoint_spec(spec, path, method)
                    methods[method] = spec
        
        return schema
    
    def _enhance_endpoint_spec(self, spec: Dict[str, Any], path: str, method: str) -> Dict[str, Any]:
        """Enhance individual endpoint specification with examples and validation."""
        # Add comprehensive examples for request body
        if 'requestBody' in spec:
            request_body = spec['requestBody']
            if 'content' in request_body:
                for content_type, content_spec in request_body['content'].items():
                    if 'schema' in content_spec and 'example' not in content_spec:
                        content_spec['example'] = self._generate_example_from_schema(
                            content_spec['schema'], content_type
                        )
        
        # Add comprehensive examples for responses
        responses = spec.get('responses', {})
        for status_code, response_spec in responses.items():
            if 'content' in response_spec:
                for content_type, content_spec in response_spec['content'].items():
                    if 'schema' in content_spec and 'example' not in content_spec:
                        content_spec['example'] = self._generate_example_from_schema(
                            content_spec['schema'], content_type
                        )
        
        # Add operation tags if missing
        if 'tags' not in spec:
            # Infer tag from path
            path_parts = path.strip('/').split('/')
            if path_parts:
                spec['tags'] = [path_parts[0].title()]
        
        return spec
    
    def _generate_example_from_schema(self, schema: Dict[str, Any], content_type: str) -> Any:
        """Generate realistic examples from OpenAPI schema."""
        schema_type = schema.get('type', 'object')
        
        if schema_type == 'object':
            example = {}
            properties = schema.get('properties', {})
            
            for prop_name, prop_schema in properties.items():
                example[prop_name] = self._generate_property_example(prop_name, prop_schema)
            
            return example
        
        elif schema_type == 'array':
            items_schema = schema.get('items', {})
            return [self._generate_example_from_schema(items_schema, content_type)]
        
        elif schema_type == 'string':
            return self._generate_string_example(schema)
        
        elif schema_type == 'integer':
            return schema.get('example', 42)
        
        elif schema_type == 'number':
            return schema.get('example', 3.14)
        
        elif schema_type == 'boolean':
            return schema.get('example', True)
        
        return None
    
    def _generate_property_example(self, prop_name: str, prop_schema: Dict[str, Any]) -> Any:
        """Generate realistic examples for object properties."""
        prop_type = prop_schema.get('type', 'string')
        
        # Use existing example if available
        if 'example' in prop_schema:
            return prop_schema['example']
        
        # Generate contextual examples based on property name
        prop_name_lower = prop_name.lower()
        
        if prop_type == 'string':
            if 'id' in prop_name_lower:
                return f"example-{prop_name_lower}-123"
            elif 'name' in prop_name_lower:
                return f"Example {prop_name.title()}"
            elif 'email' in prop_name_lower:
                return "user@example.com"
            elif 'url' in prop_name_lower or 'uri' in prop_name_lower:
                return "https://example.com/resource"
            elif 'description' in prop_name_lower:
                return f"Description for {prop_name}"
            elif 'status' in prop_name_lower:
                return "active"
            else:
                return f"example_{prop_name_lower}"
        
        elif prop_type == 'integer':
            if 'count' in prop_name_lower or 'total' in prop_name_lower:
                return 10
            elif 'age' in prop_name_lower:
                return 5
            elif 'score' in prop_name_lower:
                return 85
            else:
                return 42
        
        elif prop_type == 'number':
            if 'score' in prop_name_lower or 'rating' in prop_name_lower:
                return 4.5
            elif 'threshold' in prop_name_lower:
                return 0.95
            else:
                return 3.14
        
        elif prop_type == 'boolean':
            if 'is_' in prop_name_lower or 'has_' in prop_name_lower:
                return True
            else:
                return False
        
        elif prop_type == 'array':
            items_schema = prop_schema.get('items', {})
            return [self._generate_example_from_schema(items_schema, 'application/json')]
        
        elif prop_type == 'object':
            return self._generate_example_from_schema(prop_schema, 'application/json')
        
        return None
    
    def _generate_string_example(self, schema: Dict[str, Any]) -> str:
        """Generate string examples based on format and constraints."""
        format_type = schema.get('format')
        
        if format_type == 'date-time':
            return "2023-12-01T10:30:00Z"
        elif format_type == 'date':
            return "2023-12-01"
        elif format_type == 'time':
            return "10:30:00"
        elif format_type == 'email':
            return "user@example.com"
        elif format_type == 'uri':
            return "https://example.com/resource"
        elif format_type == 'uuid':
            return "123e4567-e89b-12d3-a456-426614174000"
        
        # Check for enum values
        if 'enum' in schema:
            return schema['enum'][0]
        
        # Use pattern or default
        return schema.get('example', 'example_string')
    
    def _save_openapi_schema(self, schema: Dict[str, Any]):
        """Save enhanced OpenAPI schema to file."""
        with open(self.api_dir / "openapi.json", "w") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
        
        # Also save as YAML for better readability
        try:
            import yaml
            with open(self.api_dir / "openapi.yaml", "w") as f:
                yaml.dump(schema, f, default_flow_style=False, allow_unicode=True)
        except ImportError:
            pass  # YAML export is optional
    
    def _generate_enhanced_endpoint_docs(self, schema: Dict[str, Any]):
        """Generate enhanced endpoint documentation with comprehensive details."""
        endpoints_dir = self.api_dir / "endpoints"
        endpoints_dir.mkdir(exist_ok=True)
        
        paths = schema.get("paths", {})
        
        for path, methods in paths.items():
            for method, spec in methods.items():
                if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                    self._create_enhanced_endpoint_doc(endpoints_dir, path, method, spec, schema)
    
    def _create_enhanced_endpoint_doc(self, endpoints_dir: Path, path: str, method: str, 
                                    spec: Dict[str, Any], full_schema: Dict[str, Any]):
        """Create comprehensive documentation for a single endpoint."""
        # Clean path for filename
        filename = f"{method.upper()}_{path.replace('/', '_').replace('{', '').replace('}', '')}.md"
        filepath = endpoints_dir / filename
        
        # Build comprehensive documentation
        content = f"""# {method.upper()} {path}

## Summary
{spec.get('summary', 'No summary available')}

## Description
{spec.get('description', 'No description available')}

## Operation ID
`{spec.get('operationId', 'N/A')}`

## Tags
{', '.join(spec.get('tags', ['Untagged']))}

"""
        
        # Add authentication requirements
        security = spec.get('security', full_schema.get('security', []))
        if security:
            content += "## Authentication\n"
            for security_req in security:
                for scheme_name, scopes in security_req.items():
                    content += f"- **{scheme_name}**: {', '.join(scopes) if scopes else 'Required'}\n"
            content += "\n"
        
        # Add parameters with enhanced details
        parameters = spec.get('parameters', [])
        if parameters:
            content += "## Parameters\n\n"
            content += "| Name | Location | Type | Required | Description |\n"
            content += "|------|----------|------|----------|-------------|\n"
            
            for param in parameters:
                param_name = param.get('name', 'unknown')
                param_in = param.get('in', 'query')
                param_required = '✓' if param.get('required', False) else '✗'
                param_desc = param.get('description', 'No description')
                param_schema = param.get('schema', {})
                param_type = param_schema.get('type', 'string')
                
                content += f"| `{param_name}` | {param_in} | {param_type} | {param_required} | {param_desc} |\n"
            
            content += "\n"
        else:
            content += "## Parameters\nNo parameters required.\n\n"
        
        # Add request body with examples
        request_body = spec.get('requestBody')
        if request_body:
            content += "## Request Body\n\n"
            content += f"**Description**: {request_body.get('description', 'Request body required')}\n\n"
            content += f"**Required**: {'Yes' if request_body.get('required', False) else 'No'}\n\n"
            
            request_content = request_body.get('content', {})
            for content_type, content_spec in request_content.items():
                content += f"### Content Type: `{content_type}`\n\n"
                
                # Add schema information
                schema_info = content_spec.get('schema', {})
                if schema_info:
                    content += "**Schema**:\n```json\n"
                    content += json.dumps(schema_info, indent=2)
                    content += "\n```\n\n"
                
                # Add example
                example = content_spec.get('example')
                if example:
                    content += "**Example**:\n```json\n"
                    content += json.dumps(example, indent=2, ensure_ascii=False)
                    content += "\n```\n\n"
        
        # Add responses with enhanced details
        responses = spec.get('responses', {})
        if responses:
            content += "## Responses\n\n"
            
            for status_code, response_spec in responses.items():
                content += f"### {status_code} - {response_spec.get('description', 'No description')}\n\n"
                
                response_content = response_spec.get('content', {})
                for content_type, content_spec in response_content.items():
                    content += f"**Content Type**: `{content_type}`\n\n"
                    
                    # Add schema
                    schema_info = content_spec.get('schema', {})
                    if schema_info:
                        content += "**Schema**:\n```json\n"
                        content += json.dumps(schema_info, indent=2)
                        content += "\n```\n\n"
                    
                    # Add example
                    example = content_spec.get('example')
                    if example:
                        content += "**Example Response**:\n```json\n"
                        content += json.dumps(example, indent=2, ensure_ascii=False)
                        content += "\n```\n\n"
                
                # Add headers if present
                headers = response_spec.get('headers', {})
                if headers:
                    content += "**Response Headers**:\n"
                    for header_name, header_spec in headers.items():
                        header_desc = header_spec.get('description', 'No description')
                        content += f"- `{header_name}`: {header_desc}\n"
                    content += "\n"
        
        # Add comprehensive examples
        content += "## Complete Example\n\n"
        content += f"### Request\n```http\n{method.upper()} {path} HTTP/1.1\n"
        content += "Host: localhost:8000\n"
        content += "Content-Type: application/json\n"
        
        # Add authentication header example if required
        if security:
            content += "Authorization: Bearer <your-token>\n"
        
        content += "\n"
        
        # Add request body example if present
        if request_body:
            request_content = request_body.get('content', {})
            json_content = request_content.get('application/json', {})
            example = json_content.get('example')
            if example:
                content += json.dumps(example, indent=2, ensure_ascii=False)
        
        content += "\n```\n\n"
        
        # Add response example
        if responses:
            success_response = responses.get('200') or responses.get('201') or list(responses.values())[0]
            content += "### Response\n```http\nHTTP/1.1 200 OK\n"
            content += "Content-Type: application/json\n\n"
            
            response_content = success_response.get('content', {})
            json_content = response_content.get('application/json', {})
            example = json_content.get('example')
            if example:
                content += json.dumps(example, indent=2, ensure_ascii=False)
            
            content += "\n```\n\n"
        
        # Add error handling section
        error_responses = {k: v for k, v in responses.items() if k.startswith('4') or k.startswith('5')}
        if error_responses:
            content += "## Error Handling\n\n"
            for status_code, error_spec in error_responses.items():
                content += f"### {status_code} Error\n"
                content += f"{error_spec.get('description', 'Error occurred')}\n\n"
        
        # Add notes and best practices
        content += "## Notes\n\n"
        content += f"- **Endpoint**: `{method.upper()} {path}`\n"
        content += f"- **Operation ID**: `{spec.get('operationId', 'N/A')}`\n"
        
        if spec.get('deprecated'):
            content += "- **⚠️ DEPRECATED**: This endpoint is deprecated and may be removed in future versions.\n"
        
        content += "\n"
        
        with open(filepath, "w", encoding='utf-8') as f:
            f.write(content)
    
    def _generate_swagger_ui(self, schema: Dict[str, Any]):
        """Generate interactive Swagger UI with custom styling."""
        swagger_dir = self.api_dir / "swagger"
        swagger_dir.mkdir(exist_ok=True)
        
        # Generate custom Swagger UI HTML
        swagger_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{schema['info']['title']} - API Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui.css" />
    <style>
        html {{
            box-sizing: border-box;
            overflow: -moz-scrollbars-vertical;
            overflow-y: scroll;
        }}
        
        *, *:before, *:after {{
            box-sizing: inherit;
        }}
        
        body {{
            margin:0;
            background: #fafafa;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        
        .swagger-ui .topbar {{
            background-color: #2c3e50;
            padding: 10px 0;
        }}
        
        .swagger-ui .topbar .download-url-wrapper {{
            display: none;
        }}
        
        .swagger-ui .info .title {{
            color: #2c3e50;
        }}
        
        .swagger-ui .scheme-container {{
            background: #fff;
            box-shadow: 0 1px 2px 0 rgba(0,0,0,.15);
        }}
        
        .custom-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
        }}
        
        .custom-header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        
        .custom-header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        
        .api-info {{
            background: white;
            padding: 20px;
            margin: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .search-container {{
            margin: 20px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .search-input {{
            width: 100%;
            padding: 10px;
            border: 2px solid #e1e5e9;
            border-radius: 4px;
            font-size: 16px;
        }}
        
        .search-input:focus {{
            outline: none;
            border-color: #667eea;
        }}
    </style>
</head>
<body>
    <div class="custom-header">
        <h1>{schema['info']['title']}</h1>
        <p>{schema['info'].get('description', 'API Documentation')}</p>
        <p>Version: {schema['info']['version']}</p>
    </div>
    
    <div class="search-container">
        <input type="text" class="search-input" placeholder="Search endpoints..." id="endpoint-search">
    </div>
    
    <div class="api-info">
        <h3>API Information</h3>
        <p><strong>Base URL:</strong> {schema.get('servers', [{}])[0].get('url', 'http://localhost:8000')}</p>
        <p><strong>OpenAPI Version:</strong> {schema['openapi']}</p>
        <p><strong>Total Endpoints:</strong> <span id="endpoint-count">{sum(len(methods) for methods in schema.get('paths', {}).values())}</span></p>
    </div>
    
    <div id="swagger-ui"></div>
    
    <script src="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {{
            const ui = SwaggerUIBundle({{
                url: '../openapi.json',
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                tryItOutEnabled: true,
                filter: true,
                supportedSubmitMethods: ['get', 'post', 'put', 'delete', 'patch'],
                onComplete: function() {{
                    console.log("Swagger UI loaded successfully");
                    setupSearch();
                }},
                requestInterceptor: function(request) {{
                    // Add custom headers or modify requests here
                    console.log("Request:", request);
                    return request;
                }},
                responseInterceptor: function(response) {{
                    // Handle responses here
                    console.log("Response:", response);
                    return response;
                }}
            }});
            
            function setupSearch() {{
                const searchInput = document.getElementById('endpoint-search');
                if (searchInput) {{
                    searchInput.addEventListener('input', function(e) {{
                        const searchTerm = e.target.value.toLowerCase();
                        const operations = document.querySelectorAll('.opblock');
                        
                        operations.forEach(function(operation) {{
                            const summary = operation.querySelector('.opblock-summary-description');
                            const path = operation.querySelector('.opblock-summary-path');
                            
                            if (summary && path) {{
                                const summaryText = summary.textContent.toLowerCase();
                                const pathText = path.textContent.toLowerCase();
                                
                                if (summaryText.includes(searchTerm) || pathText.includes(searchTerm)) {{
                                    operation.style.display = 'block';
                                }} else {{
                                    operation.style.display = 'none';
                                }}
                            }}
                        }});
                    }});
                }}
            }}
        }};
    </script>
</body>
</html>"""
        
        with open(swagger_dir / "index.html", "w", encoding='utf-8') as f:
            f.write(swagger_html)
        
        # Generate a simple redirect from api root
        api_index_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="0; url=swagger/">
    <title>Redirecting to API Documentation</title>
</head>
<body>
    <p>Redirecting to <a href="swagger/">API Documentation</a>...</p>
</body>
</html>"""
        
        with open(self.api_dir / "index.html", "w") as f:
            f.write(api_index_html)
    
    def _generate_api_overview(self, schema: Dict[str, Any]):
        """Generate comprehensive API overview documentation."""
        
        # Define the curl examples separately to avoid f-string backslash issues
        curl_examples = '''### cURL Examples
```bash
# Get API health status
curl -X GET "http://localhost:8000/api/v1/health"

# Upload a drawing for analysis
curl -X POST "http://localhost:8000/api/v1/drawings/upload" \\
  -H "Content-Type: multipart/form-data" \\
  -F "file=@drawing.png" \\
  -F "age=5"
```'''
        
        overview_content = f"""# {schema['info']['title']} - API Documentation

{schema['info'].get('description', 'API documentation for the Children\'s Drawing Anomaly Detection System')}

**Version**: {schema['info']['version']}  
**OpenAPI Version**: {schema['openapi']}  
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Quick Start

### Base URL
```
{schema.get('servers', [{}])[0].get('url', 'http://localhost:8000')}
```

### Interactive Documentation
- **Swagger UI**: [Interactive API Explorer](swagger/)
- **OpenAPI Spec**: [JSON](openapi.json) | [YAML](openapi.yaml)

## API Overview

### Endpoints Summary
"""
        
        # Group endpoints by tags
        paths = schema.get('paths', {})
        endpoints_by_tag = {}
        
        for path, methods in paths.items():
            for method, spec in methods.items():
                if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                    tags = spec.get('tags', ['Untagged'])
                    for tag in tags:
                        if tag not in endpoints_by_tag:
                            endpoints_by_tag[tag] = []
                        endpoints_by_tag[tag].append({
                            'path': path,
                            'method': method.upper(),
                            'summary': spec.get('summary', 'No summary'),
                            'operationId': spec.get('operationId', '')
                        })
        
        # Generate endpoint tables by tag
        for tag, endpoints in endpoints_by_tag.items():
            overview_content += f"\n### {tag}\n\n"
            overview_content += "| Method | Endpoint | Summary |\n"
            overview_content += "|--------|----------|----------|\n"
            
            for endpoint in endpoints:
                method_badge = f"`{endpoint['method']}`"
                overview_content += f"| {method_badge} | `{endpoint['path']}` | {endpoint['summary']} |\n"
            
            overview_content += "\n"
        
        # Add authentication section
        security_schemes = schema.get('components', {}).get('securitySchemes', {})
        if security_schemes:
            overview_content += "## Authentication\n\n"
            for scheme_name, scheme_spec in security_schemes.items():
                scheme_type = scheme_spec.get('type', 'unknown')
                overview_content += f"### {scheme_name}\n"
                overview_content += f"- **Type**: {scheme_type}\n"
                
                if 'description' in scheme_spec:
                    overview_content += f"- **Description**: {scheme_spec['description']}\n"
                
                if scheme_type == 'http':
                    overview_content += f"- **Scheme**: {scheme_spec.get('scheme', 'bearer')}\n"
                elif scheme_type == 'apiKey':
                    overview_content += f"- **Location**: {scheme_spec.get('in', 'header')}\n"
                    overview_content += f"- **Parameter**: {scheme_spec.get('name', 'X-API-Key')}\n"
                
                overview_content += "\n"
        
        # Add error handling section
        overview_content += """## Error Handling

The API uses conventional HTTP response codes to indicate success or failure:

- **2xx**: Success
- **4xx**: Client error (invalid request, authentication failure, etc.)
- **5xx**: Server error

### Common Error Responses

#### 400 Bad Request
```json
{
  "detail": "Invalid request parameters"
}
```

#### 401 Unauthorized
```json
{
  "detail": "Authentication required"
}
```

#### 404 Not Found
```json
{
  "detail": "Resource not found"
}
```

#### 422 Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "field_name"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

#### 500 Internal Server Error
```json
{
  "detail": "Internal server error"
}
```

## Rate Limiting

API requests may be subject to rate limiting. Check response headers for rate limit information:

- `X-RateLimit-Limit`: Request limit per time window
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Time when the rate limit resets

## SDKs and Tools

"""
        
        # Add the curl examples (defined separately to avoid f-string backslash issues)
        overview_content += curl_examples
        
        # Add Python examples
        overview_content += """

### Python Example
```python
import requests

# API base URL
base_url = "http://localhost:8000/api/v1"

# Upload and analyze a drawing
with open("drawing.png", "rb") as f:
    response = requests.post(
        f"{base_url}/drawings/upload",
        files={"file": f},
        data={"age": 5}
    )

if response.status_code == 200:
    result = response.json()
    print(f"Analysis ID: {result['id']}")
else:
    print(f"Error: {response.status_code}")
```

## Support

For API support and questions:
- **Documentation**: [Full API Reference](endpoints/)
- **Interactive Testing**: [Swagger UI](swagger/)
- **Issues**: Report bugs and feature requests through the project repository

---

*This documentation is automatically generated from the OpenAPI specification.*
"""
        
        with open(self.api_dir / "README.md", "w", encoding='utf-8') as f:
            f.write(overview_content)
    
    def _generate_auth_documentation(self, schema: Dict[str, Any]):
        """Generate detailed authentication documentation."""
        auth_dir = self.api_dir / "authentication"
        auth_dir.mkdir(exist_ok=True)
        
        security_schemes = schema.get('components', {}).get('securitySchemes', {})
        
        if not security_schemes:
            # Generate basic auth documentation even if no schemes defined
            auth_content = """# Authentication

This API currently does not require authentication for most endpoints. However, some administrative endpoints may require authentication in the future.

## Future Authentication Plans

The API is designed to support the following authentication methods:

### Bearer Token Authentication
- **Type**: HTTP Bearer Token
- **Header**: `Authorization: Bearer <token>`
- **Use Case**: API access tokens for programmatic access

### API Key Authentication  
- **Type**: API Key
- **Header**: `X-API-Key: <key>`
- **Use Case**: Simple API key-based access

## Implementation Status

Currently, the API operates in development mode without authentication requirements. Authentication will be implemented in future versions for:

- Administrative operations
- Rate limiting bypass
- Premium features access
- User-specific data access

## Security Considerations

Even without authentication, the API implements:
- Input validation and sanitization
- Rate limiting (planned)
- CORS protection
- Request size limits
- File type validation for uploads

For production deployment, authentication should be implemented before exposing the API publicly.
"""
        else:
            auth_content = "# Authentication\n\n"
            auth_content += "This API supports multiple authentication methods:\n\n"
            
            for scheme_name, scheme_spec in security_schemes.items():
                auth_content += f"## {scheme_name}\n\n"
                auth_content += f"{scheme_spec.get('description', 'Authentication scheme')}\n\n"
                
                scheme_type = scheme_spec.get('type', 'unknown')
                
                if scheme_type == 'http':
                    scheme_name_lower = scheme_spec.get('scheme', 'bearer').lower()
                    auth_content += f"**Type**: HTTP {scheme_name_lower.title()}\n\n"
                    
                    if scheme_name_lower == 'bearer':
                        auth_content += """**Usage**:
```http
Authorization: Bearer <your-token>
```

**Example**:
```bash
curl -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..." \\
     https://api.example.com/endpoint
```
"""
                
                elif scheme_type == 'apiKey':
                    key_location = scheme_spec.get('in', 'header')
                    key_name = scheme_spec.get('name', 'X-API-Key')
                    
                    auth_content += f"**Type**: API Key\n"
                    auth_content += f"**Location**: {key_location}\n"
                    auth_content += f"**Parameter Name**: {key_name}\n\n"
                    
                    if key_location == 'header':
                        auth_content += f"""**Usage**:
```http
{key_name}: <your-api-key>
```

**Example**:
```bash
curl -H "{key_name}: your-api-key-here" \\
     https://api.example.com/endpoint
```
"""
                    elif key_location == 'query':
                        auth_content += f"""**Usage**:
```
https://api.example.com/endpoint?{key_name}=<your-api-key>
```

**Example**:
```bash
curl "https://api.example.com/endpoint?{key_name}=your-api-key-here"
```
"""
                
                auth_content += "\n"
        
        with open(auth_dir / "README.md", "w", encoding='utf-8') as f:
            f.write(auth_content)
    
    def _generate_error_specifications(self, schema: Dict[str, Any]):
        """Generate comprehensive error specifications documentation."""
        errors_dir = self.api_dir / "errors"
        errors_dir.mkdir(exist_ok=True)
        
        # Extract error schemas from components
        components = schema.get('components', {})
        error_schemas = {}
        
        # Look for error-related schemas
        schemas = components.get('schemas', {})
        for schema_name, schema_spec in schemas.items():
            if any(keyword in schema_name.lower() for keyword in ['error', 'exception', 'validation']):
                error_schemas[schema_name] = schema_spec
        
        # Generate error documentation
        error_content = """# Error Specifications

This document provides comprehensive information about error responses from the API.

## Error Response Format

All API errors follow a consistent format to ensure predictable error handling:

```json
{
  "detail": "Human-readable error message",
  "error_code": "MACHINE_READABLE_CODE",
  "timestamp": "2023-12-01T10:30:00Z",
  "path": "/api/v1/endpoint",
  "request_id": "req_123456789"
}
```

## HTTP Status Codes

### 2xx Success
- **200 OK**: Request successful
- **201 Created**: Resource created successfully
- **202 Accepted**: Request accepted for processing
- **204 No Content**: Request successful, no content returned

### 4xx Client Errors
- **400 Bad Request**: Invalid request syntax or parameters
- **401 Unauthorized**: Authentication required or invalid
- **403 Forbidden**: Access denied (valid auth but insufficient permissions)
- **404 Not Found**: Requested resource not found
- **405 Method Not Allowed**: HTTP method not supported for endpoint
- **409 Conflict**: Request conflicts with current resource state
- **422 Unprocessable Entity**: Request syntax valid but semantically incorrect
- **429 Too Many Requests**: Rate limit exceeded

### 5xx Server Errors
- **500 Internal Server Error**: Unexpected server error
- **502 Bad Gateway**: Invalid response from upstream server
- **503 Service Unavailable**: Server temporarily unavailable
- **504 Gateway Timeout**: Upstream server timeout

## Common Error Scenarios

### Validation Errors (422)

When request data fails validation:

```json
{
  "detail": [
    {
      "loc": ["body", "age"],
      "msg": "ensure this value is greater than 0",
      "type": "value_error.number.not_gt",
      "ctx": {"limit_value": 0}
    },
    {
      "loc": ["body", "file"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**Fields**:
- `loc`: Location of the error (path to the field)
- `msg`: Human-readable error message
- `type`: Machine-readable error type
- `ctx`: Additional context (when available)

### File Upload Errors

#### Invalid File Type (400)
```json
{
  "detail": "Invalid file type. Supported formats: PNG, JPEG, BMP",
  "error_code": "INVALID_FILE_TYPE",
  "supported_formats": ["image/png", "image/jpeg", "image/bmp"]
}
```

#### File Too Large (413)
```json
{
  "detail": "File size exceeds maximum limit of 10MB",
  "error_code": "FILE_TOO_LARGE",
  "max_size_bytes": 10485760,
  "received_size_bytes": 15728640
}
```

### Authentication Errors

#### Missing Authentication (401)
```json
{
  "detail": "Authentication credentials were not provided",
  "error_code": "AUTHENTICATION_REQUIRED"
}
```

#### Invalid Token (401)
```json
{
  "detail": "Invalid or expired authentication token",
  "error_code": "INVALID_TOKEN"
}
```

### Resource Errors

#### Not Found (404)
```json
{
  "detail": "Drawing with ID 'abc123' not found",
  "error_code": "RESOURCE_NOT_FOUND",
  "resource_type": "drawing",
  "resource_id": "abc123"
}
```

#### Conflict (409)
```json
{
  "detail": "Drawing with this filename already exists",
  "error_code": "RESOURCE_CONFLICT",
  "conflicting_field": "filename"
}
```

### Rate Limiting (429)

```json
{
  "detail": "Rate limit exceeded. Try again in 60 seconds",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "retry_after": 60,
  "limit": 100,
  "window": "1h"
}
```

### Server Errors (5xx)

#### Internal Server Error (500)
```json
{
  "detail": "An unexpected error occurred. Please try again later",
  "error_code": "INTERNAL_SERVER_ERROR",
  "request_id": "req_123456789"
}
```

#### Service Unavailable (503)
```json
{
  "detail": "ML model service temporarily unavailable",
  "error_code": "SERVICE_UNAVAILABLE",
  "service": "ml_inference",
  "retry_after": 30
}
```

## Error Handling Best Practices

### Client Implementation

1. **Always check HTTP status codes** before processing response body
2. **Parse error details** from the response body for user-friendly messages
3. **Implement retry logic** for 5xx errors and 429 rate limiting
4. **Log error details** including request_id for debugging

### Example Error Handling (Python)

```python
import requests
import time

def api_request_with_retry(url, **kwargs):
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            response = requests.request(**kwargs, url=url)
            
            if response.status_code < 400:
                return response.json()
            
            elif response.status_code == 429:
                # Rate limited - check retry-after header
                retry_after = int(response.headers.get('Retry-After', retry_delay))
                time.sleep(retry_after)
                continue
            
            elif 500 <= response.status_code < 600:
                # Server error - retry with exponential backoff
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                    continue
            
            # Client error or final attempt - raise exception
            error_data = response.json()
            raise APIError(
                status_code=response.status_code,
                detail=error_data.get('detail', 'Unknown error'),
                error_code=error_data.get('error_code')
            )
            
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
                continue
            raise

class APIError(Exception):
    def __init__(self, status_code, detail, error_code=None):
        self.status_code = status_code
        self.detail = detail
        self.error_code = error_code
        super().__init__(f"{status_code}: {detail}")
```

## Error Code Reference

| Error Code | HTTP Status | Description |
|------------|-------------|-------------|
| `VALIDATION_ERROR` | 422 | Request data validation failed |
| `INVALID_FILE_TYPE` | 400 | Unsupported file format |
| `FILE_TOO_LARGE` | 413 | File exceeds size limit |
| `AUTHENTICATION_REQUIRED` | 401 | Missing authentication |
| `INVALID_TOKEN` | 401 | Invalid or expired token |
| `INSUFFICIENT_PERMISSIONS` | 403 | Access denied |
| `RESOURCE_NOT_FOUND` | 404 | Requested resource not found |
| `RESOURCE_CONFLICT` | 409 | Resource already exists |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_SERVER_ERROR` | 500 | Unexpected server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily down |
| `ML_MODEL_ERROR` | 500 | ML inference failed |
| `DATABASE_ERROR` | 500 | Database operation failed |

---

*For additional support, include the `request_id` from error responses when reporting issues.*
"""
        
        # Add schema-specific error documentation if available
        if error_schemas:
            error_content += "\n## Error Schema Definitions\n\n"
            for schema_name, schema_spec in error_schemas.items():
                error_content += f"### {schema_name}\n\n"
                error_content += "```json\n"
                error_content += json.dumps(schema_spec, indent=2)
                error_content += "\n```\n\n"
        
        with open(errors_dir / "README.md", "w", encoding='utf-8') as f:
            f.write(error_content)


class InterfaceGenerator:
    """
    UML 2.5 compliant interface documentation generator.
    
    Generates comprehensive interface documentation including service contracts,
    sequence diagrams, class diagrams, and component diagrams with validation
    against actual implementation.
    """
    
    def __init__(self, project_root: Path, docs_dir: Path):
        self.project_root = project_root
        self.docs_dir = docs_dir
        self.app_dir = project_root / "app"
        self.interfaces_dir = docs_dir / "interfaces"
        self.interfaces_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different interface documentation types
        (self.interfaces_dir / "services").mkdir(exist_ok=True)
        (self.interfaces_dir / "uml").mkdir(exist_ok=True)
        (self.interfaces_dir / "uml" / "sequences").mkdir(exist_ok=True)
        (self.interfaces_dir / "uml" / "classes").mkdir(exist_ok=True)
        (self.interfaces_dir / "uml" / "components").mkdir(exist_ok=True)
        (self.interfaces_dir / "contracts").mkdir(exist_ok=True)
        (self.interfaces_dir / "dtos").mkdir(exist_ok=True)
    
    def generate_service_contracts(self) -> List[Dict[str, Any]]:
        """
        Generate UML 2.5 compliant service interface contracts from Python type hints.
        
        Extracts service interfaces from type annotations and generates formal
        contract specifications with method signatures, parameters, and return types.
        
        Returns:
            List of service contract specifications
        """
        contracts = []
        
        if not self.app_dir.exists():
            return contracts
        
        # Analyze service files for interface contracts
        services_dir = self.app_dir / "services"
        if services_dir.exists():
            for service_file in services_dir.glob("*.py"):
                if service_file.name != "__init__.py":
                    contract = self._extract_service_contract(service_file)
                    if contract:
                        contracts.append(contract)
                        self._generate_contract_documentation(contract)
        
        # Analyze API endpoints for interface contracts
        api_dir = self.app_dir / "api"
        if api_dir.exists():
            endpoints_dir = api_dir / "api_v1" / "endpoints"
            if endpoints_dir.exists():
                for endpoint_file in endpoints_dir.glob("*.py"):
                    if endpoint_file.name != "__init__.py":
                        contract = self._extract_api_contract(endpoint_file)
                        if contract:
                            contracts.append(contract)
                            self._generate_contract_documentation(contract)
        
        return contracts
    
    def _extract_service_contract(self, service_file: Path) -> Optional[Dict[str, Any]]:
        """Extract service contract from Python service file."""
        try:
            content = service_file.read_text()
            tree = ast.parse(content)
            
            contract = {
                'name': service_file.stem.replace('_', ' ').title(),
                'file': str(service_file.relative_to(self.project_root)),
                'type': 'service',
                'classes': [],
                'interfaces': [],
                'methods': [],
                'dependencies': []
            }
            
            # Extract imports for dependency analysis
            imports = self._extract_imports(tree)
            contract['dependencies'] = [imp for imp in imports.values() 
                                      if 'app.services' in imp or 'app.models' in imp]
            
            # Extract classes and their interfaces
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self._analyze_class_interface(node, service_file)
                    contract['classes'].append(class_info)
                    
                    # Extract public methods as interface
                    public_methods = [m for m in class_info['methods'] 
                                    if not m['name'].startswith('_')]
                    if public_methods:
                        interface = {
                            'name': f"{class_info['name']}Interface",
                            'class': class_info['name'],
                            'methods': public_methods,
                            'type': 'protocol'
                        }
                        contract['interfaces'].append(interface)
                        contract['methods'].extend(public_methods)
            
            return contract if contract['classes'] else None
            
        except (SyntaxError, UnicodeDecodeError, IOError):
            return None
    
    def _extract_api_contract(self, endpoint_file: Path) -> Optional[Dict[str, Any]]:
        """Extract API contract from endpoint file."""
        try:
            content = endpoint_file.read_text()
            tree = ast.parse(content)
            
            contract = {
                'name': f"{endpoint_file.stem.title()} API",
                'file': str(endpoint_file.relative_to(self.project_root)),
                'type': 'api',
                'endpoints': [],
                'methods': [],
                'dependencies': []
            }
            
            # Extract FastAPI route definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Look for FastAPI decorators
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call):
                            if hasattr(decorator.func, 'attr') and decorator.func.attr in ['get', 'post', 'put', 'delete']:
                                endpoint_info = self._analyze_endpoint_interface(node, decorator)
                                contract['endpoints'].append(endpoint_info)
                                contract['methods'].append(endpoint_info)
            
            return contract if contract['endpoints'] else None
            
        except (SyntaxError, UnicodeDecodeError, IOError):
            return None
    
    def _analyze_class_interface(self, class_node: ast.ClassDef, file_path: Path) -> Dict[str, Any]:
        """Analyze a class definition for interface information."""
        class_info = {
            'name': class_node.name,
            'file': str(file_path.relative_to(self.project_root)),
            'docstring': ast.get_docstring(class_node) or "",
            'methods': [],
            'attributes': [],
            'base_classes': [],
            'is_abstract': False
        }
        
        # Extract base classes
        for base in class_node.bases:
            if isinstance(base, ast.Name):
                class_info['base_classes'].append(base.id)
                if base.id in ['ABC', 'Protocol']:
                    class_info['is_abstract'] = True
        
        # Extract methods with type annotations
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._analyze_method_interface(item)
                class_info['methods'].append(method_info)
                
                # Check for abstract methods
                for decorator in item.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == 'abstractmethod':
                        method_info['is_abstract'] = True
                        class_info['is_abstract'] = True
            
            elif isinstance(item, ast.AnnAssign):
                # Extract annotated attributes
                if isinstance(item.target, ast.Name):
                    attr_info = {
                        'name': item.target.id,
                        'type_annotation': self._get_annotation_string(item.annotation),
                        'line_number': item.lineno
                    }
                    class_info['attributes'].append(attr_info)
        
        return class_info
    
    def _analyze_method_interface(self, method_node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze a method definition for interface information."""
        method_info = {
            'name': method_node.name,
            'docstring': ast.get_docstring(method_node) or "",
            'parameters': [],
            'return_annotation': None,
            'is_abstract': False,
            'is_property': False,
            'visibility': 'public' if not method_node.name.startswith('_') else 'private'
        }
        
        # Extract parameters with type annotations
        for arg in method_node.args.args:
            if arg.arg != 'self':  # Skip self parameter
                param_info = {
                    'name': arg.arg,
                    'type_annotation': self._get_annotation_string(arg.annotation) if arg.annotation else None
                }
                method_info['parameters'].append(param_info)
        
        # Extract return annotation
        if method_node.returns:
            method_info['return_annotation'] = self._get_annotation_string(method_node.returns)
        
        # Check for property decorator
        for decorator in method_node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == 'property':
                method_info['is_property'] = True
        
        return method_info
    
    def _analyze_endpoint_interface(self, func_node: ast.FunctionDef, decorator: ast.Call) -> Dict[str, Any]:
        """Analyze FastAPI endpoint for interface information."""
        endpoint_info = {
            'name': func_node.name,
            'docstring': ast.get_docstring(func_node) or "",
            'method': 'GET',  # Default
            'path': '/',
            'parameters': [],
            'return_annotation': None
        }
        
        # Extract HTTP method from decorator
        if hasattr(decorator.func, 'attr'):
            endpoint_info['method'] = decorator.func.attr.upper()
        
        # Extract path from decorator arguments
        if decorator.args and isinstance(decorator.args[0], ast.Constant):
            endpoint_info['path'] = decorator.args[0].value
        
        # Extract parameters
        for arg in func_node.args.args:
            param_info = {
                'name': arg.arg,
                'type_annotation': self._get_annotation_string(arg.annotation) if arg.annotation else None
            }
            endpoint_info['parameters'].append(param_info)
        
        # Extract return annotation
        if func_node.returns:
            endpoint_info['return_annotation'] = self._get_annotation_string(func_node.returns)
        
        return endpoint_info
    
    def _get_annotation_string(self, annotation) -> str:
        """Convert AST annotation to string representation."""
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Constant):
            return str(annotation.value)
        elif isinstance(annotation, ast.Attribute):
            return f"{annotation.value.id}.{annotation.attr}"
        elif isinstance(annotation, ast.Subscript):
            # Handle generic types like List[str], Dict[str, Any]
            if isinstance(annotation.value, ast.Name):
                base_type = annotation.value.id
                if isinstance(annotation.slice, ast.Name):
                    return f"{base_type}[{annotation.slice.id}]"
                elif isinstance(annotation.slice, ast.Tuple):
                    args = []
                    for elt in annotation.slice.elts:
                        if isinstance(elt, ast.Name):
                            args.append(elt.id)
                        else:
                            args.append(str(elt))
                    return f"{base_type}[{', '.join(args)}]"
        return str(annotation)
    
    def _extract_imports(self, tree: ast.AST) -> Dict[str, str]:
        """Extract import statements for dependency analysis."""
        imports = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        full_name = f"{node.module}.{alias.name}"
                        imports[alias.asname or alias.name] = full_name
        
        return imports
    
    def _generate_contract_documentation(self, contract: Dict[str, Any]):
        """Generate documentation file for a service contract."""
        contract_name = contract['name'].lower().replace(' ', '-')
        contract_file = self.interfaces_dir / "services" / f"{contract_name}-contract.md"
        
        content = f"""# {contract['name']} Contract

## Overview
Service contract for {contract['name']} ({contract['type']})

**Source File**: `{contract['file']}`

## Interface Specification

"""
        
        # Add class interfaces
        if contract.get('classes'):
            content += "### Classes\n\n"
            for class_info in contract['classes']:
                content += f"#### {class_info['name']}\n\n"
                if class_info['docstring']:
                    content += f"{class_info['docstring']}\n\n"
                
                if class_info['base_classes']:
                    content += f"**Inherits from**: {', '.join(class_info['base_classes'])}\n\n"
                
                if class_info['is_abstract']:
                    content += "**Type**: Abstract Class\n\n"
                
                # Add attributes
                if class_info['attributes']:
                    content += "**Attributes**:\n\n"
                    for attr in class_info['attributes']:
                        type_str = f": {attr['type_annotation']}" if attr['type_annotation'] else ""
                        content += f"- `{attr['name']}{type_str}`\n"
                    content += "\n"
        
        # Add methods section
        if contract.get('methods'):
            content += "## Methods\n\n"
            for method in contract['methods']:
                content += f"### {method['name']}\n\n"
                
                if method['docstring']:
                    content += f"{method['docstring']}\n\n"
                
                # Method signature
                params = []
                for param in method['parameters']:
                    param_str = param['name']
                    if param['type_annotation']:
                        param_str += f": {param['type_annotation']}"
                    params.append(param_str)
                
                signature = f"{method['name']}({', '.join(params)})"
                if method['return_annotation']:
                    signature += f" -> {method['return_annotation']}"
                
                content += f"**Signature**: `{signature}`\n\n"
                
                if method.get('is_abstract'):
                    content += "**Type**: Abstract Method\n\n"
                if method.get('is_property'):
                    content += "**Type**: Property\n\n"
                
                # Parameters table
                if method['parameters']:
                    content += "**Parameters**:\n\n"
                    content += "| Name | Type | Description |\n"
                    content += "|------|------|-------------|\n"
                    for param in method['parameters']:
                        param_type = param['type_annotation'] or 'Any'
                        content += f"| `{param['name']}` | `{param_type}` | Parameter description |\n"
                    content += "\n"
                
                if method['return_annotation']:
                    content += f"**Returns**: `{method['return_annotation']}`\n\n"
        
        # Add dependencies
        if contract.get('dependencies'):
            content += "## Dependencies\n\n"
            for dep in contract['dependencies']:
                content += f"- `{dep}`\n"
            content += "\n"
        
        # Add interfaces
        if contract.get('interfaces'):
            content += "## Defined Interfaces\n\n"
            for interface in contract['interfaces']:
                content += f"### {interface['name']}\n\n"
                content += f"**Type**: {interface['type'].title()}\n"
                content += f"**Implemented by**: {interface['class']}\n\n"
                
                if interface['methods']:
                    content += "**Methods**:\n\n"
                    for method in interface['methods']:
                        params = [f"{p['name']}: {p['type_annotation'] or 'Any'}" 
                                for p in method['parameters']]
                        signature = f"{method['name']}({', '.join(params)})"
                        if method['return_annotation']:
                            signature += f" -> {method['return_annotation']}"
                        content += f"- `{signature}`\n"
                    content += "\n"
        
        content += """## Usage Examples

```python
# Example usage of the service contract
# TODO: Add specific usage examples
```

## Validation

This contract is automatically validated against the implementation in:
- Source file: `{}`
- Last validated: {}

""".format(contract['file'], datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        with open(contract_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def generate_sequence_diagram(self, interaction: str) -> Dict[str, Any]:
        """
        Generate sequence diagrams showing interaction flows between system components.
        
        Args:
            interaction: Name of the interaction to document
            
        Returns:
            Dictionary containing sequence diagram information
        """
        # Analyze service interactions to generate sequence diagrams
        interactions = self._discover_service_interactions()
        
        for interaction_info in interactions:
            self._generate_sequence_diagram_file(interaction_info)
        
        return {'interactions': interactions}
    
    def _discover_service_interactions(self) -> List[Dict[str, Any]]:
        """Discover service interactions from codebase analysis."""
        interactions = []
        
        # Analyze API endpoints for interaction patterns
        api_dir = self.app_dir / "api"
        if api_dir.exists():
            endpoints_dir = api_dir / "api_v1" / "endpoints"
            if endpoints_dir.exists():
                for endpoint_file in endpoints_dir.glob("*.py"):
                    if endpoint_file.name != "__init__.py":
                        endpoint_interactions = self._analyze_endpoint_interactions(endpoint_file)
                        interactions.extend(endpoint_interactions)
        
        return interactions
    
    def _analyze_endpoint_interactions(self, endpoint_file: Path) -> List[Dict[str, Any]]:
        """Analyze endpoint file for service interaction patterns."""
        interactions = []
        
        try:
            content = endpoint_file.read_text()
            tree = ast.parse(content)
            
            # Look for function calls that indicate service interactions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if this is an API endpoint
                    is_endpoint = any(
                        isinstance(d, ast.Call) and 
                        hasattr(d.func, 'attr') and 
                        d.func.attr in ['get', 'post', 'put', 'delete']
                        for d in node.decorator_list
                    )
                    
                    if is_endpoint:
                        interaction = self._extract_interaction_from_function(node, endpoint_file)
                        if interaction:
                            interactions.append(interaction)
        
        except (SyntaxError, UnicodeDecodeError, IOError):
            pass
        
        return interactions
    
    def _extract_interaction_from_function(self, func_node: ast.FunctionDef, file_path: Path) -> Optional[Dict[str, Any]]:
        """Extract interaction pattern from function definition."""
        interaction = {
            'name': f"{file_path.stem}_{func_node.name}",
            'title': f"{func_node.name.replace('_', ' ').title()} Interaction",
            'participants': ['Client'],
            'messages': [],
            'source_file': str(file_path.relative_to(self.project_root))
        }
        
        # Add API endpoint as participant
        endpoint_name = f"{file_path.stem.title()}API"
        interaction['participants'].append(endpoint_name)
        
        # Analyze function body for service calls
        service_calls = []
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    # Look for service method calls
                    if hasattr(node.func.value, 'id') and 'service' in node.func.value.id.lower():
                        service_name = node.func.value.id
                        method_name = node.func.attr
                        service_calls.append({
                            'service': service_name,
                            'method': method_name
                        })
                        
                        # Add service as participant
                        service_title = service_name.replace('_', ' ').title()
                        if service_title not in interaction['participants']:
                            interaction['participants'].append(service_title)
        
        # Generate message sequence
        interaction['messages'].append({
            'from': 'Client',
            'to': endpoint_name,
            'message': f"{func_node.name}()",
            'type': 'request'
        })
        
        for call in service_calls:
            service_title = call['service'].replace('_', ' ').title()
            interaction['messages'].append({
                'from': endpoint_name,
                'to': service_title,
                'message': f"{call['method']}()",
                'type': 'call'
            })
            
            interaction['messages'].append({
                'from': service_title,
                'to': endpoint_name,
                'message': 'result',
                'type': 'return'
            })
        
        interaction['messages'].append({
            'from': endpoint_name,
            'to': 'Client',
            'message': 'response',
            'type': 'response'
        })
        
        return interaction if len(interaction['participants']) > 2 else None
    
    def _generate_sequence_diagram_file(self, interaction: Dict[str, Any]):
        """Generate sequence diagram file for an interaction."""
        diagram_file = self.interfaces_dir / "uml" / "sequences" / f"{interaction['name']}.md"
        
        content = f"""# {interaction['title']}

## Overview
Sequence diagram showing the interaction flow for {interaction['title'].lower()}.

**Source**: `{interaction['source_file']}`

## Participants
"""
        
        for participant in interaction['participants']:
            content += f"- **{participant}**\n"
        
        content += "\n## Sequence Diagram\n\n```mermaid\nsequenceDiagram\n"
        
        # Add participants
        for participant in interaction['participants']:
            content += f"    participant {participant.replace(' ', '')}\n"
        
        content += "\n"
        
        # Add messages
        for msg in interaction['messages']:
            from_participant = msg['from'].replace(' ', '')
            to_participant = msg['to'].replace(' ', '')
            message = msg['message']
            
            if msg['type'] == 'return':
                content += f"    {to_participant}-->{from_participant}: {message}\n"
            else:
                content += f"    {from_participant}->{to_participant}: {message}\n"
        
        content += "```\n\n"
        
        content += """## Message Details

| From | To | Message | Type | Description |
|------|----|---------|----- |-------------|
"""
        
        for msg in interaction['messages']:
            content += f"| {msg['from']} | {msg['to']} | `{msg['message']}` | {msg['type']} | Message description |\n"
        
        content += "\n"
        
        with open(diagram_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def generate_class_diagram(self, module: str = None) -> Dict[str, Any]:
        """
        Generate class diagrams for data model relationships.
        
        Args:
            module: Optional specific module to focus on
            
        Returns:
            Dictionary containing class diagram information
        """
        class_info = self._analyze_class_relationships()
        
        # Generate class diagram files
        for module_name, classes in class_info.items():
            self._generate_class_diagram_file(module_name, classes)
        
        return class_info
    
    def _analyze_class_relationships(self) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze class relationships across the codebase."""
        class_info = {}
        
        # Analyze service classes
        services_dir = self.app_dir / "services"
        if services_dir.exists():
            service_classes = []
            for service_file in services_dir.glob("*.py"):
                if service_file.name != "__init__.py":
                    classes = self._extract_classes_from_file(service_file)
                    service_classes.extend(classes)
            
            if service_classes:
                class_info['services'] = service_classes
        
        # Analyze model classes
        models_dir = self.app_dir / "models"
        if models_dir.exists():
            model_classes = []
            for model_file in models_dir.glob("*.py"):
                if model_file.name != "__init__.py":
                    classes = self._extract_classes_from_file(model_file)
                    model_classes.extend(classes)
            
            if model_classes:
                class_info['models'] = model_classes
        
        return class_info
    
    def _extract_classes_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract class information from a Python file."""
        classes = []
        
        try:
            content = file_path.read_text()
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'file': str(file_path.relative_to(self.project_root)),
                        'base_classes': [],
                        'methods': [],
                        'attributes': [],
                        'relationships': []
                    }
                    
                    # Extract base classes
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            class_info['base_classes'].append(base.id)
                            class_info['relationships'].append({
                                'type': 'inheritance',
                                'target': base.id,
                                'description': f"{node.name} inherits from {base.id}"
                            })
                    
                    # Extract methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_info = {
                                'name': item.name,
                                'visibility': 'public' if not item.name.startswith('_') else 'private'
                            }
                            class_info['methods'].append(method_info)
                        
                        elif isinstance(item, ast.AnnAssign):
                            if isinstance(item.target, ast.Name):
                                attr_info = {
                                    'name': item.target.id,
                                    'type': self._get_annotation_string(item.annotation) if item.annotation else 'Any'
                                }
                                class_info['attributes'].append(attr_info)
                    
                    classes.append(class_info)
        
        except (SyntaxError, UnicodeDecodeError, IOError):
            pass
        
        return classes
    
    def _generate_class_diagram_file(self, module_name: str, classes: List[Dict[str, Any]]):
        """Generate class diagram file for a module."""
        diagram_file = self.interfaces_dir / "uml" / "classes" / f"{module_name}-classes.md"
        
        content = f"""# {module_name.title()} Class Diagram

## Overview
Class diagram showing the structure and relationships of {module_name} classes.

## Classes Overview

"""
        
        for class_info in classes:
            content += f"### {class_info['name']}\n"
            content += f"**Source**: `{class_info['file']}`\n\n"
            
            if class_info['base_classes']:
                content += f"**Inherits from**: {', '.join(class_info['base_classes'])}\n\n"
            
            if class_info['attributes']:
                content += "**Attributes**:\n"
                for attr in class_info['attributes']:
                    content += f"- `{attr['name']}: {attr['type']}`\n"
                content += "\n"
            
            if class_info['methods']:
                public_methods = [m for m in class_info['methods'] if m['visibility'] == 'public']
                if public_methods:
                    content += "**Public Methods**:\n"
                    for method in public_methods:
                        content += f"- `{method['name']}()`\n"
                    content += "\n"
        
        content += "## Class Diagram\n\n```mermaid\nclassDiagram\n"
        
        # Add class definitions
        for class_info in classes:
            content += f"    class {class_info['name']} {{\n"
            
            # Add attributes
            for attr in class_info['attributes']:
                content += f"        +{attr['type']} {attr['name']}\n"
            
            # Add public methods
            public_methods = [m for m in class_info['methods'] if m['visibility'] == 'public']
            for method in public_methods:
                content += f"        +{method['name']}()\n"
            
            content += "    }\n\n"
        
        # Add relationships
        for class_info in classes:
            for relationship in class_info['relationships']:
                if relationship['type'] == 'inheritance':
                    content += f"    {relationship['target']} <|-- {class_info['name']}\n"
        
        content += "```\n\n"
        
        # Add relationships table
        all_relationships = []
        for class_info in classes:
            all_relationships.extend(class_info['relationships'])
        
        if all_relationships:
            content += "## Relationships\n\n"
            content += "| From | To | Type | Description |\n"
            content += "|------|----|----- |-------------|\n"
            
            for class_info in classes:
                for rel in class_info['relationships']:
                    content += f"| {class_info['name']} | {rel['target']} | {rel['type']} | {rel['description']} |\n"
            
            content += "\n"
        
        with open(diagram_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def generate_component_diagram(self) -> Dict[str, Any]:
        """
        Generate component diagrams for system architecture.
        
        Returns:
            Dictionary containing component diagram information
        """
        components = self._analyze_system_components()
        self._generate_component_diagram_file(components)
        
        return components
    
    def _analyze_system_components(self) -> Dict[str, Any]:
        """Analyze system components and their relationships."""
        components = {
            'components': [],
            'relationships': [],
            'interfaces': []
        }
        
        # Analyze API layer components
        api_dir = self.app_dir / "api"
        if api_dir.exists():
            api_component = {
                'name': 'API Layer',
                'type': 'component',
                'description': 'REST API endpoints and request handling',
                'interfaces': ['HTTP REST API'],
                'dependencies': ['Service Layer']
            }
            components['components'].append(api_component)
        
        # Analyze service layer components
        services_dir = self.app_dir / "services"
        if services_dir.exists():
            service_files = list(services_dir.glob("*.py"))
            if service_files:
                service_component = {
                    'name': 'Service Layer',
                    'type': 'component',
                    'description': 'Business logic and service implementations',
                    'interfaces': ['Service Interfaces'],
                    'dependencies': ['Data Layer']
                }
                components['components'].append(service_component)
        
        # Analyze data layer components
        models_dir = self.app_dir / "models"
        if models_dir.exists():
            data_component = {
                'name': 'Data Layer',
                'type': 'component',
                'description': 'Data models and database access',
                'interfaces': ['Data Access Interface'],
                'dependencies': ['Database']
            }
            components['components'].append(data_component)
        
        # Add relationships
        for component in components['components']:
            for dep in component.get('dependencies', []):
                components['relationships'].append({
                    'from': component['name'],
                    'to': dep,
                    'type': 'depends_on'
                })
        
        return components
    
    def _generate_component_diagram_file(self, components: Dict[str, Any]):
        """Generate component diagram file."""
        diagram_file = self.interfaces_dir / "uml" / "components" / "system-components.md"
        
        content = """# System Component Diagram

## Overview
Component diagram showing the high-level system architecture and component relationships.

## Components

"""
        
        for component in components['components']:
            content += f"### {component['name']}\n"
            content += f"**Type**: {component['type'].title()}\n"
            content += f"**Description**: {component['description']}\n\n"
            
            if component.get('interfaces'):
                content += "**Provides Interfaces**:\n"
                for interface in component['interfaces']:
                    content += f"- {interface}\n"
                content += "\n"
            
            if component.get('dependencies'):
                content += "**Dependencies**:\n"
                for dep in component['dependencies']:
                    content += f"- {dep}\n"
                content += "\n"
        
        content += "## Component Diagram\n\n```mermaid\ngraph TB\n"
        
        # Add components
        for component in components['components']:
            comp_id = component['name'].replace(' ', '')
            content += f"    {comp_id}[{component['name']}]\n"
        
        content += "\n"
        
        # Add relationships
        for relationship in components['relationships']:
            from_id = relationship['from'].replace(' ', '')
            to_id = relationship['to'].replace(' ', '')
            content += f"    {from_id} --> {to_id}\n"
        
        content += "```\n\n"
        
        # Add relationships table
        if components['relationships']:
            content += "## Component Relationships\n\n"
            content += "| From | To | Type | Description |\n"
            content += "|------|----|----- |-------------|\n"
            
            for rel in components['relationships']:
                content += f"| {rel['from']} | {rel['to']} | {rel['type']} | Component dependency |\n"
            
            content += "\n"
        
        with open(diagram_file, 'w', encoding='utf-8') as f:
            f.write(content)


class DocumentationEngine:
    """
    Enhanced documentation generation engine with comprehensive automation and validation.
    
    This class serves as the main orchestrator for all documentation generation activities,
    providing change detection, incremental updates, and dependency management.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.docs_dir = project_root / "docs"
        self.app_dir = project_root / "app"
        self.frontend_dir = project_root / "frontend"
        self.cache_dir = project_root / ".kiro" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize change detection and dependency management
        self.change_detector = ChangeDetector(self.cache_dir / "file_cache.json")
        self.dependency_manager = DependencyGraph()
        self.doc_cache = DocumentationCache(self.cache_dir / "doc_cache.json")
        self._setup_dependencies()
        
        # Initialize architecture generator
        self.architecture_generator = ArchitectureGenerator(project_root)
        
        # Initialize comprehensive validation engine
        self.validation_engine = ComprehensiveValidationEngine(project_root, self.docs_dir)
        
        # Initialize search and navigation engines
        self.search_engine = DocumentationSearchEngine(project_root)
        self.navigation_engine = NavigationEngine(project_root)
        
        # Track generation state
        self.last_generation_time = datetime.now()
        self.generation_history: List[GenerationResult] = []
    
    def _setup_dependencies(self):
        """Set up dependency relationships between documentation components."""
        # API documentation depends on service implementations
        self.dependency_manager.add_dependency("api", "services")
        self.dependency_manager.add_dependency("api", "database")
        
        # Algorithm documentation depends on service implementations
        self.dependency_manager.add_dependency("algorithms", "services")
        
        # Interface documentation depends on multiple components
        self.dependency_manager.add_dependency("interfaces", "services")
        self.dependency_manager.add_dependency("interfaces", "api")
        
        # Architecture documentation depends on all components
        for doc_type in [DocumentationType.API, DocumentationType.SERVICES, 
                        DocumentationType.DATABASE, DocumentationType.FRONTEND]:
            self.dependency_manager.add_dependency("architecture", doc_type.value)
        
        # Workflow documentation depends on multiple components
        self.dependency_manager.add_dependency("workflows", "services")
        self.dependency_manager.add_dependency("workflows", "api")
        self.dependency_manager.add_dependency("workflows", "frontend")
        
        # Set up file patterns for change detection
        self.dependency_manager.add_file_pattern("api", "api/")
        self.dependency_manager.add_file_pattern("api", "endpoints/")
        self.dependency_manager.add_file_pattern("services", "services/")
        self.dependency_manager.add_file_pattern("algorithms", "services/")
        self.dependency_manager.add_file_pattern("database", "models/")
        self.dependency_manager.add_file_pattern("database", "alembic/")
        self.dependency_manager.add_file_pattern("frontend", "frontend/")
        self.dependency_manager.add_file_pattern("frontend", ".tsx")
        self.dependency_manager.add_file_pattern("frontend", ".ts")
        self.dependency_manager.add_file_pattern("workflows", "services/")
        self.dependency_manager.add_file_pattern("workflows", "api/")
        self.dependency_manager.add_file_pattern("workflows", "frontend/")
        
    def generate_all(self, force: bool = False) -> GenerationResult:
        """
        Generate all documentation categories with change detection.
        
        Args:
            force: If True, regenerate all documentation regardless of changes
            
        Returns:
            GenerationResult with details of the generation process
        """
        start_time = time.time()
        result = GenerationResult(success=True)
        
        print("🚀 Starting comprehensive documentation generation...")
        
        try:
            # Detect changes in source files
            source_paths = [self.app_dir, self.frontend_dir]
            changes = self.change_detector.detect_changes(source_paths)
            result.changes_detected = changes
            
            if not force and not changes:
                print("📋 No changes detected, skipping generation")
                result.duration = time.time() - start_time
                return result
            
            # Determine affected documentation types
            affected_types = self.dependency_manager.get_affected_components(changes)
            
            if force:
                # Generate all documentation types
                affected_types = set(DocumentationType)
                # Invalidate all cache entries when forcing regeneration
                for doc_type in DocumentationType:
                    self.doc_cache.invalidate_component(doc_type.value)
            
            # Filter out components that are still cached and valid
            if not force:
                affected_types = self._filter_cached_components(affected_types, changes)
            
            if not affected_types:
                print("📋 All documentation is up to date (using cache)")
                result.duration = time.time() - start_time
                return result
            
            # Get generation order based on dependencies
            generation_order = self.dependency_manager.get_generation_order(affected_types)
            
            print(f"📝 Generating documentation for: {[t.value for t in generation_order]}")
            
            # Generate documentation in dependency order
            for doc_type in generation_order:
                category_result = self.generate_category(doc_type)
                result.generated_files.extend(category_result.generated_files)
                result.errors.extend(category_result.errors)
                result.warnings.extend(category_result.warnings)
                
                # Update cache for successfully generated components
                if category_result.success and category_result.generated_files:
                    source_hash = self._calculate_source_hash(doc_type)
                    cache_entry = CacheEntry(
                        content_hash=source_hash,
                        generated_files=category_result.generated_files,
                        timestamp=datetime.now(),
                        dependencies=self.dependency_manager.dependencies.get(doc_type.value, set())
                    )
                    self.doc_cache.set_entry(doc_type.value, cache_entry)
            
            # Always update the documentation index
            self.update_documentation_index()
            result.generated_files.append(self.docs_dir / "README.md")
            
            # Update search index and navigation structure
            print("🔍 Updating search index and navigation...")
            self._update_search_and_navigation()
            
            # Generate enhanced navigation files
            self._generate_navigation_files()
            result.generated_files.extend(self._get_navigation_files())
            
            # Validate generated documentation with comprehensive validation
            print("🔍 Running comprehensive validation...")
            validation_result = self.validate_sources()
            
            if validation_result.is_valid:
                print(f"  ✅ All documentation passed validation ({validation_result.metrics.files_validated} files)")
            else:
                print(f"  ⚠️  Validation found {len(validation_result.errors)} errors and {len(validation_result.warnings)} warnings")
                
                # Add validation results to generation result
                for error in validation_result.errors:
                    result.errors.append(f"Validation error in {error.file_path}: {error.message}")
                
                for warning in validation_result.warnings:
                    result.warnings.append(f"Validation warning in {warning.file_path}: {warning.message}")
            
            # Add validation recommendations
            if validation_result.recommendations:
                result.warnings.extend([f"Recommendation: {rec}" for rec in validation_result.recommendations[:5]])
            
            self.last_generation_time = datetime.now()
            
        except Exception as e:
            result.success = False
            result.errors.append(f"Generation failed: {str(e)}")
            print(f"❌ Documentation generation failed: {e}")
        
        result.duration = time.time() - start_time
        self.generation_history.append(result)
        
        if result.success:
            print(f"✅ Documentation generation complete! ({result.duration:.2f}s)")
        
        return result
    
    def generate_category(self, category: DocumentationType) -> GenerationResult:
        """
        Generate documentation for a specific category.
        
        Args:
            category: The type of documentation to generate
            
        Returns:
            GenerationResult for the specific category
        """
        result = GenerationResult(success=True)
        
        try:
            if category == DocumentationType.API:
                self.generate_api_docs()
                result.generated_files.extend(self._get_generated_files(self.docs_dir / "api"))
            elif category == DocumentationType.SERVICES:
                self.generate_service_docs()
                result.generated_files.extend(self._get_generated_files(self.docs_dir / "interfaces" / "services"))
            elif category == DocumentationType.ALGORITHMS:
                self.generate_algorithm_docs()
                result.generated_files.extend(self._get_generated_files(self.docs_dir / "algorithms" / "implementations"))
            elif category == DocumentationType.DATABASE:
                self.generate_database_docs()
                result.generated_files.extend(self._get_generated_files(self.docs_dir / "database"))
            elif category == DocumentationType.FRONTEND:
                self.generate_frontend_docs()
                result.generated_files.extend(self._get_generated_files(self.docs_dir / "frontend"))
            elif category == DocumentationType.DEPLOYMENT:
                self.generate_deployment_docs()
                result.generated_files.extend(self._get_generated_files(self.docs_dir / "deployment"))
            elif category == DocumentationType.ARCHITECTURE:
                self.generate_architecture_docs()
                result.generated_files.extend(self._get_generated_files(self.docs_dir / "architecture"))
            elif category == DocumentationType.WORKFLOWS:
                self.generate_workflow_docs()
                result.generated_files.extend(self._get_generated_files(self.docs_dir / "workflows"))
                
        except Exception as e:
            result.success = False
            result.errors.append(f"Failed to generate {category.value} documentation: {str(e)}")
        
        return result
    
    def _get_generated_files(self, directory: Path) -> List[Path]:
        """Get list of generated files in a directory."""
        if not directory.exists():
            return []
        return list(directory.rglob("*.md"))
    
    def validate_sources(self) -> ValidationResult:
        """
        Validate all generated documentation sources using comprehensive validation.
        
        Returns:
            ValidationResult with comprehensive validation details
        """
        try:
            # Use comprehensive validation engine for thorough validation
            import asyncio
            
            # Run comprehensive validation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                comprehensive_result = loop.run_until_complete(
                    self.validation_engine.validate_comprehensive()
                )
                return comprehensive_result
            finally:
                loop.close()
                
        except Exception as e:
            # Fallback to basic validation if comprehensive validation fails
            print(f"⚠️  Comprehensive validation failed, falling back to basic validation: {e}")
            return self._basic_validate_sources()
    
    def _basic_validate_sources(self) -> ValidationResult:
        """
        Basic validation fallback method.
        
        Returns:
            ValidationResult with basic validation details
        """
        from scripts.validation_engine import ValidationResult as BasicValidationResult
        result = BasicValidationResult(is_valid=True)
        
        try:
            # Validate all markdown files in docs directory
            for md_file in self.docs_dir.rglob("*.md"):
                file_result = self._validate_markdown_file(md_file)
                result.validated_files.append(md_file)
                
                if not file_result['valid']:
                    result.is_valid = False
                    # Convert basic errors to ValidationError objects
                    from scripts.validation_engine import ValidationError
                    for error_msg in file_result['errors']:
                        result.errors.append(ValidationError(
                            file_path=md_file,
                            line_number=None,
                            error_type="basic_validation_error",
                            message=error_msg
                        ))
                
                # Convert basic warnings to ValidationWarning objects
                from scripts.validation_engine import ValidationWarning
                for warning_msg in file_result['warnings']:
                    result.warnings.append(ValidationWarning(
                        file_path=md_file,
                        line_number=None,
                        warning_type="basic_validation_warning",
                        message=warning_msg
                    ))
                
        except Exception as e:
            from scripts.validation_engine import ValidationError
            result.is_valid = False
            result.errors.append(ValidationError(
                file_path=Path("unknown"),
                line_number=None,
                error_type="validation_system_error",
                message=f"Basic validation failed: {str(e)}"
            ))
        
        return result
    
    def _validate_markdown_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate a single markdown file."""
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            content = file_path.read_text()
            
            # Basic validation checks
            if len(content.strip()) == 0:
                result['valid'] = False
                result['errors'].append(f"Empty file: {file_path}")
            
            # Check for proper markdown structure
            lines = content.split('\n')
            has_title = any(line.startswith('# ') for line in lines)
            
            if not has_title:
                result['warnings'].append(f"No main title found: {file_path}")
            
            # Check for broken internal links (basic check)
            import re
            internal_links = re.findall(r'\[.*?\]\((?!http)([^)]+)\)', content)
            for link in internal_links:
                link_path = file_path.parent / link
                if not link_path.exists() and not (self.docs_dir / link).exists():
                    result['warnings'].append(f"Potentially broken link '{link}' in {file_path}")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Failed to validate {file_path}: {str(e)}")
        
        return result
    
    def detect_changes(self) -> List[FileChange]:
        """
        Detect changes in source files since last generation.
        
        Returns:
            List of detected file changes
        """
        source_paths = [self.app_dir, self.frontend_dir]
        return self.change_detector.detect_changes(source_paths)
    
    def _filter_cached_components(self, affected_types: Set[DocumentationType], 
                                 changes: List[FileChange]) -> Set[DocumentationType]:
        """
        Filter out components that are still cached and valid.
        
        Args:
            affected_types: Set of potentially affected documentation types
            changes: List of file changes detected
            
        Returns:
            Set of documentation types that actually need regeneration
        """
        needs_regeneration = set()
        
        for doc_type in affected_types:
            source_hash = self._calculate_source_hash(doc_type)
            
            if not self.doc_cache.is_valid(doc_type.value, source_hash):
                needs_regeneration.add(doc_type)
                print(f"🔄 {doc_type.value} needs regeneration (cache invalid)")
            else:
                print(f"✅ {doc_type.value} is up to date (using cache)")
        
        return needs_regeneration
    
    def _update_search_and_navigation(self):
        """Update search index and navigation structure."""
        try:
            # Index documentation for search
            search_result = self.search_engine.index_documentation()
            print(f"  📚 Indexed {search_result['indexed']} documents for search")
            
            # Build navigation structure
            nav_result = self.navigation_engine.build_navigation_structure()
            print(f"  🧭 Built navigation for {nav_result['total_documents']} documents")
            
        except Exception as e:
            print(f"  ⚠️  Error updating search/navigation: {e}")
    
    def _generate_navigation_files(self):
        """Generate navigation-related files."""
        try:
            # Generate sitemap
            sitemap = self.navigation_engine.generate_sitemap()
            sitemap_file = self.docs_dir / "sitemap.json"
            with open(sitemap_file, 'w') as f:
                json.dump(sitemap, f, indent=2)
            
            # Generate cross-reference report
            xref_report = self.navigation_engine.generate_cross_reference_report()
            xref_file = self.docs_dir / "cross-references.json"
            with open(xref_file, 'w') as f:
                json.dump(xref_report, f, indent=2)
            
            # Generate search statistics
            search_stats = self.search_engine.get_statistics()
            stats_file = self.docs_dir / "search-statistics.json"
            with open(stats_file, 'w') as f:
                json.dump(search_stats, f, indent=2)
            
            print("  📊 Generated navigation and search metadata files")
            
        except Exception as e:
            print(f"  ⚠️  Error generating navigation files: {e}")
    
    def _get_navigation_files(self) -> List[Path]:
        """Get list of generated navigation files."""
        nav_files = []
        
        for filename in ["sitemap.json", "cross-references.json", "search-statistics.json"]:
            file_path = self.docs_dir / filename
            if file_path.exists():
                nav_files.append(file_path)
        
        return nav_files
    
    def _calculate_source_hash(self, doc_type: DocumentationType) -> str:
        """
        Calculate a hash of source files that affect a documentation type.
        
        Args:
            doc_type: The documentation type to calculate hash for
            
        Returns:
            MD5 hash of relevant source files
        """
        hasher = hashlib.md5()
        
        # Get relevant source paths based on documentation type
        source_paths = []
        
        if doc_type == DocumentationType.API:
            source_paths.extend(self.app_dir.glob("api/**/*.py"))
            source_paths.extend(self.app_dir.glob("schemas/**/*.py"))
        elif doc_type == DocumentationType.SERVICES:
            source_paths.extend(self.app_dir.glob("services/**/*.py"))
        elif doc_type == DocumentationType.ALGORITHMS:
            source_paths.extend(self.app_dir.glob("services/**/*.py"))
        elif doc_type == DocumentationType.DATABASE:
            source_paths.extend(self.app_dir.glob("models/**/*.py"))
            source_paths.extend(Path("alembic").glob("versions/**/*.py") if Path("alembic").exists() else [])
        elif doc_type == DocumentationType.FRONTEND:
            if self.frontend_dir.exists():
                source_paths.extend(self.frontend_dir.glob("src/**/*.tsx"))
                source_paths.extend(self.frontend_dir.glob("src/**/*.ts"))
        
        # Sort paths for consistent hashing
        source_paths = sorted([p for p in source_paths if p.is_file()])
        
        for path in source_paths:
            try:
                hasher.update(path.read_bytes())
            except (IOError, OSError):
                # Skip files that can't be read
                pass
        
        return hasher.hexdigest()
    
    def clear_cache(self):
        """Clear all cached documentation."""
        self.doc_cache.cache.clear()
        self.doc_cache._save_cache()
        print("🗑️  Documentation cache cleared")
    
    def get_cache_status(self) -> Dict[str, Any]:
        """
        Get status of documentation cache.
        
        Returns:
            Dictionary with cache status information
        """
        status = {
            'cached_components': list(self.doc_cache.cache.keys()),
            'cache_entries': len(self.doc_cache.cache),
            'total_cached_files': sum(len(entry.generated_files) for entry in self.doc_cache.cache.values())
        }
        
        # Check validity of cached entries
        valid_entries = 0
        for component, entry in self.doc_cache.cache.items():
            try:
                doc_type = DocumentationType(component)
                source_hash = self._calculate_source_hash(doc_type)
                if self.doc_cache.is_valid(component, source_hash):
                    valid_entries += 1
            except ValueError:
                pass
        
        status['valid_entries'] = valid_entries
        status['invalid_entries'] = len(self.doc_cache.cache) - valid_entries
        
        return status
    
    def generate_api_docs(self):
        """Generate comprehensive API documentation from FastAPI OpenAPI schema."""
        print("📡 Generating API documentation...")
        
        try:
            # Import the enhanced API documentation generator
            from scripts.api_documentation_generator import APIDocumentationGenerator
            
            # Create enhanced generator instance
            api_generator = APIDocumentationGenerator(self.project_root)
            
            # Generate comprehensive API documentation
            result = api_generator.generate_enhanced_documentation()
            
            if result['success']:
                print(f"  ✅ Enhanced API documentation generated ({len(result['generated_files'])} files)")
                
                # Also generate legacy endpoint docs for backward compatibility
                try:
                    from app.main import app
                    openapi_schema = app.openapi()
                    self.generate_endpoint_docs(openapi_schema)
                except Exception as legacy_error:
                    print(f"  ⚠️  Legacy endpoint generation failed: {legacy_error}")
            else:
                print(f"  ❌ Enhanced API documentation generation failed")
                for error in result['errors']:
                    print(f"    - {error}")
                
                # Fallback to basic generation
                self._generate_basic_api_docs()
            
        except Exception as e:
            print(f"  ❌ Failed to generate enhanced API docs: {e}")
            # Fallback to basic generation
            self._generate_basic_api_docs()
    
    def _generate_basic_api_docs(self):
        """Fallback method for basic API documentation generation."""
        try:
            # Import FastAPI app to get OpenAPI schema
            from app.main import app
            
            openapi_schema = app.openapi()
            
            # Save OpenAPI schema
            api_dir = self.docs_dir / "api"
            api_dir.mkdir(exist_ok=True)
            
            with open(api_dir / "openapi.json", "w") as f:
                json.dump(openapi_schema, f, indent=2)
            
            # Generate endpoint documentation
            self.generate_endpoint_docs(openapi_schema)
            
            print("  ✅ Basic API documentation generated")
            
        except Exception as e:
            print(f"  ❌ Failed to generate basic API docs: {e}")
    
    def generate_endpoint_docs(self, openapi_schema: Dict):
        """Generate individual endpoint documentation."""
        api_dir = self.docs_dir / "api" / "endpoints"
        api_dir.mkdir(exist_ok=True)
        
        paths = openapi_schema.get("paths", {})
        
        for path, methods in paths.items():
            for method, spec in methods.items():
                if method.upper() in ["GET", "POST", "PUT", "DELETE"]:
                    self.create_endpoint_doc(api_dir, path, method, spec)
    
    def create_endpoint_doc(self, api_dir: Path, path: str, method: str, spec: Dict):
        """Create documentation for a single endpoint."""
        # Clean path for filename
        filename = f"{method.upper()}_{path.replace('/', '_').replace('{', '').replace('}', '')}.md"
        filepath = api_dir / filename
        
        content = f"""# {method.upper()} {path}

## Summary
{spec.get('summary', 'No summary available')}

## Description
{spec.get('description', 'No description available')}

## Parameters
"""
        
        # Add parameters
        parameters = spec.get('parameters', [])
        if parameters:
            for param in parameters:
                content += f"- **{param['name']}** ({param['in']}): {param.get('description', 'No description')}\n"
        else:
            content += "No parameters\n"
        
        # Add request body
        request_body = spec.get('requestBody')
        if request_body:
            content += f"\n## Request Body\n"
            content += f"{request_body.get('description', 'Request body required')}\n"
        
        # Add responses
        responses = spec.get('responses', {})
        content += f"\n## Responses\n"
        for status_code, response in responses.items():
            content += f"- **{status_code}**: {response.get('description', 'No description')}\n"
        
        # Add example
        content += f"\n## Example\n```http\n{method.upper()} {path}\n```\n"
        
        with open(filepath, "w") as f:
            f.write(content)
    
    def generate_service_docs(self):
        """Generate service interface documentation."""
        print("🔧 Generating service documentation...")
        
        services_dir = self.app_dir / "services"
        docs_services_dir = self.docs_dir / "interfaces" / "services"
        docs_services_dir.mkdir(parents=True, exist_ok=True)
        
        for service_file in services_dir.glob("*.py"):
            if service_file.name != "__init__.py":
                self.generate_service_doc(service_file, docs_services_dir)
        
        print("  ✅ Service documentation generated")
    
    def generate_service_doc(self, service_file: Path, output_dir: Path):
        """Generate documentation for a single service."""
        try:
            with open(service_file, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            doc_content = f"# {service_file.stem.title().replace('_', ' ')} Service\n\n"
            
            # Extract module docstring
            if ast.get_docstring(tree):
                doc_content += f"{ast.get_docstring(tree)}\n\n"
            
            # Extract classes and their methods
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    doc_content += f"## Class: {node.name}\n\n"
                    
                    if ast.get_docstring(node):
                        doc_content += f"{ast.get_docstring(node)}\n\n"
                    
                    # Extract methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and not item.name.startswith('_'):
                            doc_content += f"### {item.name}\n\n"
                            
                            if ast.get_docstring(item):
                                doc_content += f"{ast.get_docstring(item)}\n\n"
                            
                            # Extract function signature
                            args = [arg.arg for arg in item.args.args if arg.arg != 'self']
                            doc_content += f"**Signature**: `{item.name}({', '.join(args)})`\n\n"
            
            output_file = output_dir / f"{service_file.stem}.md"
            with open(output_file, 'w') as f:
                f.write(doc_content)
                
        except Exception as e:
            print(f"  ⚠️  Failed to generate docs for {service_file.name}: {e}")
    
    def generate_interface_docs(self):
        """Generate comprehensive UML 2.5 compliant interface documentation."""
        print("🔗 Generating interface documentation...")
        
        try:
            # Create interface generator instance
            interface_generator = InterfaceGenerator(self.project_root, self.docs_dir)
            
            # Generate service contracts (Requirement 5.1)
            print("  📋 Generating service contracts...")
            contracts = interface_generator.generate_service_contracts()
            print(f"    ✅ Generated {len(contracts)} service contracts")
            
            # Generate sequence diagrams (Requirement 5.2)
            print("  🔄 Generating sequence diagrams...")
            sequences = interface_generator.generate_sequence_diagram("all")
            sequence_count = len(sequences.get('interactions', []))
            print(f"    ✅ Generated {sequence_count} sequence diagrams")
            
            # Generate class diagrams (Requirement 5.3)
            print("  📊 Generating class diagrams...")
            class_diagrams = interface_generator.generate_class_diagram()
            class_count = sum(len(classes) for classes in class_diagrams.values())
            print(f"    ✅ Generated class diagrams for {class_count} classes")
            
            # Generate component diagrams (Requirement 5.4)
            print("  🏗️  Generating component diagrams...")
            components = interface_generator.generate_component_diagram()
            component_count = len(components.get('components', []))
            print(f"    ✅ Generated component diagram with {component_count} components")
            
            print("  ✅ Interface documentation generated successfully")
            
        except Exception as e:
            print(f"  ❌ Failed to generate interface documentation: {e}")
            # Don't raise the exception to allow other documentation to continue
    
    def generate_algorithm_docs(self):
        """Generate comprehensive IEEE-compliant algorithm documentation."""
        print("🧮 Generating comprehensive algorithm documentation...")
        
        try:
            # Import the enhanced algorithm generator
            from scripts.algorithm_documentation_generator import AlgorithmGenerator
            
            # Create algorithm generator instance
            algorithm_generator = AlgorithmGenerator(self.project_root, self.docs_dir)
            
            # Generate comprehensive documentation
            results = algorithm_generator.generate_comprehensive_algorithm_docs()
            
            # Report results
            print(f"  ✅ Algorithm documentation generated:")
            print(f"     - Algorithms documented: {results['algorithms_documented']}")
            print(f"     - Mathematical formulations: {results['mathematical_formulations']}")
            print(f"     - Complexity analyses: {results['complexity_analyses']}")
            print(f"     - Files generated: {results['files_generated']}")
            
            if results['errors']:
                print(f"     - Errors encountered: {len(results['errors'])}")
                for error in results['errors'][:3]:  # Show first 3 errors
                    print(f"       ⚠️ {error}")
                if len(results['errors']) > 3:
                    print(f"       ... and {len(results['errors']) - 3} more errors")
            
            return results
            
        except ImportError as e:
            print(f"  ⚠️  Failed to import AlgorithmGenerator: {e}")
            # Fallback to basic algorithm documentation
            self._generate_basic_algorithm_docs()
        except Exception as e:
            print(f"  ⚠️  Failed to generate comprehensive algorithm docs: {e}")
            print("  📝 Falling back to basic algorithm documentation...")
            self._generate_basic_algorithm_docs()
    
    def _generate_basic_algorithm_docs(self):
        """Generate basic algorithm documentation as fallback."""
        algorithms_dir = self.docs_dir / "algorithms" / "implementations"
        algorithms_dir.mkdir(parents=True, exist_ok=True)
        
        # Key algorithm files to document
        algorithm_files = [
            "score_normalizer.py",
            "threshold_manager.py",
            "embedding_service.py",
            "model_manager.py",
            "interpretability_engine.py"
        ]
        
        for filename in algorithm_files:
            service_file = self.app_dir / "services" / filename
            if service_file.exists():
                self._generate_basic_algorithm_doc(service_file, algorithms_dir)
        
        print("  ✅ Basic algorithm documentation generated")
    
    def _generate_basic_algorithm_doc(self, service_file: Path, output_dir: Path):
        """Generate basic algorithm-specific documentation."""
        try:
            with open(service_file, 'r') as f:
                content = f.read()
            
            # Extract algorithm-specific information
            doc_content = f"# {service_file.stem.title().replace('_', ' ')} Algorithm Implementation\n\n"
            doc_content += f"**Source**: `{service_file.relative_to(self.project_root)}`\n\n"
            doc_content += f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d')}\n\n"
            
            tree = ast.parse(content)
            
            # Extract module docstring
            if ast.get_docstring(tree):
                doc_content += f"## Overview\n\n{ast.get_docstring(tree)}\n\n"
            
            # Extract key algorithms (methods with detailed docstrings)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            docstring = ast.get_docstring(item)
                            if docstring and len(docstring) > 100:  # Detailed docstrings
                                doc_content += f"## Algorithm: {item.name}\n\n"
                                doc_content += f"{docstring}\n\n"
            
            output_file = output_dir / f"{service_file.stem}.md"
            with open(output_file, 'w') as f:
                f.write(doc_content)
                
        except Exception as e:
            print(f"  ⚠️  Failed to generate algorithm docs for {service_file.name}: {e}")
            # Fallback to basic algorithm documentation
            self._generate_basic_algorithm_docs()
    
    def _generate_basic_algorithm_docs(self):
        """Fallback method for basic algorithm documentation."""
        print("  📝 Falling back to basic algorithm documentation...")
        
        algorithms_dir = self.docs_dir / "algorithms" / "implementations"
        algorithms_dir.mkdir(parents=True, exist_ok=True)
        
        # Key algorithm files to document
        algorithm_files = [
            "score_normalizer.py",
            "threshold_manager.py",
            "embedding_service.py",
            "model_manager.py",
            "interpretability_engine.py"
        ]
        
        for filename in algorithm_files:
            service_file = self.app_dir / "services" / filename
            if service_file.exists():
                self._generate_basic_algorithm_doc(service_file, algorithms_dir)
        
        print("  ✅ Basic algorithm documentation generated")
    
    def _generate_basic_algorithm_doc(self, service_file: Path, output_dir: Path):
        """Generate basic algorithm-specific documentation."""
        try:
            with open(service_file, 'r') as f:
                content = f.read()
            
            # Extract algorithm-specific information
            doc_content = f"# {service_file.stem.title().replace('_', ' ')} Algorithm Implementation\n\n"
            doc_content += f"**Source**: `{service_file.relative_to(self.project_root)}`\n\n"
            doc_content += f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d')}\n\n"
            
            tree = ast.parse(content)
            
            # Extract module docstring
            if ast.get_docstring(tree):
                doc_content += f"## Overview\n\n{ast.get_docstring(tree)}\n\n"
            
            # Extract key algorithms (methods with detailed docstrings)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            docstring = ast.get_docstring(item)
                            if docstring and len(docstring) > 100:  # Detailed docstrings
                                doc_content += f"## Algorithm: {item.name}\n\n"
                                doc_content += f"{docstring}\n\n"
            
            output_file = output_dir / f"{service_file.stem}.md"
            with open(output_file, 'w') as f:
                f.write(doc_content)
                
        except Exception as e:
            print(f"  ⚠️  Failed to generate basic algorithm docs for {service_file.name}: {e}")
    
    def generate_database_docs(self):
        """Generate database schema documentation."""
        print("🗄️  Generating database documentation...")
        
        try:
            from app.models.database import Base
            from sqlalchemy import inspect
            
            db_dir = self.docs_dir / "database"
            db_dir.mkdir(exist_ok=True)
            
            # Generate schema documentation
            schema_content = "# Database Schema\n\n"
            schema_content += "This document describes the database schema for the Children's Drawing Anomaly Detection System.\n\n"
            
            # Document each table
            for table_name, table in Base.metadata.tables.items():
                schema_content += f"## Table: {table_name}\n\n"
                
                # Table description from model docstring if available
                schema_content += f"**Purpose**: Data storage for {table_name.replace('_', ' ')}\n\n"
                
                # Columns
                schema_content += "### Columns\n\n"
                schema_content += "| Column | Type | Nullable | Default | Description |\n"
                schema_content += "|--------|------|----------|---------|-------------|\n"
                
                for column in table.columns:
                    nullable = "Yes" if column.nullable else "No"
                    default = str(column.default.arg) if column.default else "None"
                    schema_content += f"| {column.name} | {column.type} | {nullable} | {default} | - |\n"
                
                schema_content += "\n"
                
                # Indexes
                if table.indexes:
                    schema_content += "### Indexes\n\n"
                    for index in table.indexes:
                        schema_content += f"- **{index.name}**: {', '.join([col.name for col in index.columns])}\n"
                    schema_content += "\n"
            
            with open(db_dir / "schema.md", 'w') as f:
                f.write(schema_content)
            
            print("  ✅ Database documentation generated")
            
        except Exception as e:
            print(f"  ❌ Failed to generate database docs: {e}")
    
    def generate_frontend_docs(self):
        """Generate frontend component documentation."""
        print("🎨 Generating frontend documentation...")
        
        if not self.frontend_dir.exists():
            print("  ⚠️  Frontend directory not found, skipping")
            return
        
        frontend_docs_dir = self.docs_dir / "frontend"
        frontend_docs_dir.mkdir(exist_ok=True)
        
        # Generate component documentation
        components_dir = self.frontend_dir / "src" / "pages"
        if components_dir.exists():
            self.generate_component_docs(components_dir, frontend_docs_dir)
        
        print("  ✅ Frontend documentation generated")
    
    def generate_component_docs(self, components_dir: Path, output_dir: Path):
        """Generate documentation for React components."""
        component_docs = "# Frontend Components\n\n"
        component_docs += "This document describes the React components in the application.\n\n"
        
        for component_file in components_dir.glob("*.tsx"):
            component_docs += f"## {component_file.stem}\n\n"
            component_docs += f"**File**: `{component_file.relative_to(self.frontend_dir)}`\n\n"
            
            try:
                with open(component_file, 'r') as f:
                    content = f.read()
                
                # Extract component description from comments
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if '/**' in line and i < 10:  # Look for JSDoc comments near top
                        comment_lines = []
                        j = i
                        while j < len(lines) and '*/' not in lines[j]:
                            comment_lines.append(lines[j].strip(' */'))
                            j += 1
                        if comment_lines:
                            component_docs += f"**Description**: {' '.join(comment_lines).strip()}\n\n"
                        break
                
                # Extract props interface if present
                if 'interface' in content and 'Props' in content:
                    component_docs += "**Props**: See TypeScript interface in source file\n\n"
                
            except Exception as e:
                component_docs += f"**Error**: Could not parse component file: {e}\n\n"
        
        with open(output_dir / "components.md", 'w') as f:
            f.write(component_docs)
    
    def generate_deployment_docs(self):
        """Generate deployment documentation."""
        print("🚀 Generating deployment documentation...")
        
        deployment_dir = self.docs_dir / "deployment"
        deployment_dir.mkdir(exist_ok=True)
        
        # Generate Docker documentation
        if (self.project_root / "docker-compose.yml").exists():
            self.generate_docker_docs(deployment_dir)
        
        # Generate environment setup documentation
        self.generate_environment_docs(deployment_dir)
        
        print("  ✅ Deployment documentation generated")
    
    def generate_docker_docs(self, output_dir: Path):
        """Generate Docker deployment documentation."""
        docker_content = """# Docker Deployment

## Overview
This application can be deployed using Docker and Docker Compose for easy setup and consistent environments.

## Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+

## Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd children-drawing-anomaly-detection

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Services
The Docker Compose configuration includes the following services:

### Backend Service
- **Image**: Custom Python application
- **Port**: 8000
- **Dependencies**: Database, file storage

### Frontend Service  
- **Image**: Custom React application
- **Port**: 3000
- **Dependencies**: Backend service

### Database Service
- **Image**: SQLite (file-based)
- **Storage**: Persistent volume

## Configuration
Environment variables can be configured in `.env` file:

```bash
# Copy example configuration
cp .env.example .env

# Edit configuration
nano .env
```

## Production Deployment
For production deployment, consider:

1. **Security**: Enable HTTPS, configure firewalls
2. **Scaling**: Use container orchestration (Kubernetes)
3. **Monitoring**: Add logging and monitoring solutions
4. **Backup**: Implement data backup strategies
"""
        
        with open(output_dir / "docker.md", 'w') as f:
            f.write(docker_content)
    
    def generate_environment_docs(self, output_dir: Path):
        """Generate environment setup documentation."""
        env_content = """# Environment Setup

## Development Environment

### Prerequisites
- Python 3.11+
- Node.js 18+
- Git

### Backend Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements-dev.txt

# Run database migrations
alembic upgrade head

# Start development server
uvicorn app.main:app --reload
```

### Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## Production Environment

### System Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ recommended
- **Storage**: 50GB+ for models and data
- **OS**: Linux (Ubuntu 20.04+ recommended)

### Installation
```bash
# Install system dependencies
sudo apt update
sudo apt install python3.11 python3.11-venv nodejs npm

# Clone and setup application
git clone <repository-url>
cd children-drawing-anomaly-detection
./setup.sh
```

### Configuration
1. Copy environment configuration: `cp .env.example .env`
2. Edit configuration file: `nano .env`
3. Configure database settings
4. Set up file storage paths
5. Configure ML model paths

### Service Management
```bash
# Start services
sudo systemctl start cdads-backend
sudo systemctl start cdads-frontend

# Enable auto-start
sudo systemctl enable cdads-backend
sudo systemctl enable cdads-frontend

# Check status
sudo systemctl status cdads-backend
```
"""
        
        with open(output_dir / "environment-setup.md", 'w') as f:
            f.write(env_content)
    
    def generate_workflow_docs(self):
        """Generate BPMN 2.0 compliant workflow documentation."""
        print("🔄 Generating workflow documentation...")
        
        try:
            # Create workflow generator instance
            workflow_generator = WorkflowGenerator(str(self.project_root))
            
            # Generate all workflows
            workflows = workflow_generator.generate_all_workflows()
            
            # Create workflows directory
            workflows_dir = self.docs_dir / "workflows"
            workflows_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate documentation files
            generated_files = workflow_generator.generate_workflow_documentation(str(workflows_dir))
            
            # Generate workflow overview
            self._generate_workflow_overview(workflows, workflows_dir)
            
            # Report results
            workflow_counts = {}
            for workflow in workflows.values():
                wf_type = workflow.workflow_type
                workflow_counts[wf_type] = workflow_counts.get(wf_type, 0) + 1
            
            print(f"  ✅ Workflow documentation generated:")
            print(f"     - Total workflows: {len(workflows)}")
            for wf_type, count in workflow_counts.items():
                print(f"     - {wf_type.replace('_', ' ').title()}: {count}")
            print(f"     - Files generated: {len(generated_files)}")
            
            return {
                'workflows_documented': len(workflows),
                'workflow_types': list(workflow_counts.keys()),
                'files_generated': len(generated_files),
                'generated_files': list(generated_files.keys())
            }
            
        except Exception as e:
            print(f"  ⚠️  Failed to generate workflow documentation: {e}")
            # Fallback to basic workflow documentation
            self._generate_basic_workflow_docs()
            return {
                'workflows_documented': 0,
                'workflow_types': [],
                'files_generated': 0,
                'generated_files': [],
                'error': str(e)
            }
    
    def _generate_workflow_overview(self, workflows: dict, output_dir: Path):
        """Generate workflow overview documentation."""
        overview_content = """# Workflow Documentation

This directory contains comprehensive workflow documentation for the Children's Drawing Anomaly Detection System.

## Overview

The system implements several types of workflows to support different aspects of the drawing analysis process:

"""
        
        # Group workflows by type
        workflow_types = {}
        for workflow in workflows.values():
            wf_type = workflow.workflow_type
            if wf_type not in workflow_types:
                workflow_types[wf_type] = []
            workflow_types[wf_type].append(workflow)
        
        # Document each workflow type
        for wf_type, wf_list in workflow_types.items():
            type_title = wf_type.replace('_', ' ').title()
            overview_content += f"## {type_title}\n\n"
            
            if wf_type == "user_journey":
                overview_content += "User journey workflows document the end-to-end user experience and interactions with the system.\n\n"
            elif wf_type == "technical_process":
                overview_content += "Technical process workflows document the internal system processes and service interactions.\n\n"
            elif wf_type == "integration_flow":
                overview_content += "Integration workflows document external system interactions and API integrations.\n\n"
            elif wf_type == "error_flow":
                overview_content += "Error handling workflows document exception handling and recovery processes.\n\n"
            elif wf_type == "ml_pipeline":
                overview_content += "Machine learning pipeline workflows document the ML training and inference processes.\n\n"
            
            for workflow in wf_list:
                overview_content += f"- **[{workflow.name}]({workflow.workflow_id}.md)**: {workflow.description}\n"
            
            overview_content += "\n"
        
        overview_content += """## BPMN 2.0 Compliance

All workflow diagrams are generated in BPMN 2.0 format and include:

- **Process Elements**: Start events, end events, tasks, gateways
- **Flow Elements**: Sequence flows connecting process elements
- **Artifacts**: Data objects and annotations where applicable
- **Swimlanes**: Organizational units and responsibilities

## Workflow Types

### User Journey Workflows
Document user interactions and system responses for different user personas.

### Technical Process Workflows  
Document internal system processes, service interactions, and data flows.

### Integration Flow Workflows
Document external system integrations, API calls, and data exchange patterns.

### Error Flow Workflows
Document exception handling, error recovery, and escalation procedures.

### ML Pipeline Workflows
Document machine learning training, inference, and model management processes.

## Usage

Each workflow includes:
- Markdown documentation with detailed descriptions
- BPMN XML files for import into BPMN modeling tools
- Process element details and properties
- Flow relationships and dependencies

---

*This documentation is automatically generated from source code analysis.*
"""
        
        overview_file = output_dir / "README.md"
        overview_file.write_text(overview_content, encoding='utf-8')
    
    def _generate_basic_workflow_docs(self):
        """Generate basic workflow documentation as fallback."""
        workflows_dir = self.docs_dir / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)
        
        # Create basic workflow documentation
        basic_content = """# Workflow Documentation

## Drawing Upload and Analysis Workflow

1. **User Upload**: User uploads drawing through web interface
2. **Validation**: System validates file format and size
3. **Processing**: Image is processed and features extracted
4. **Analysis**: ML model analyzes drawing for anomalies
5. **Results**: Anomaly score and visualizations generated
6. **Storage**: Results stored in database

## Model Training Workflow

1. **Data Collection**: Gather training drawings
2. **Preprocessing**: Clean and prepare data
3. **Feature Extraction**: Extract ViT embeddings
4. **Model Training**: Train autoencoder models
5. **Validation**: Evaluate model performance
6. **Deployment**: Deploy trained models

---

*Basic workflow documentation generated as fallback.*
"""
        
        basic_file = workflows_dir / "README.md"
        basic_file.write_text(basic_content, encoding='utf-8')
        print("  ✅ Basic workflow documentation generated")

    def generate_architecture_docs(self):
        """Generate C4 Model architecture documentation."""
        print("🏗️  Generating architecture documentation...")
        
        architecture_dir = self.docs_dir / "architecture"
        architecture_dir.mkdir(exist_ok=True)
        
        try:
            # Generate C4 Level 1: System Context
            system_context = self.architecture_generator.generate_system_context()
            self._create_system_context_doc(architecture_dir, system_context)
            
            # Generate C4 Level 2: Container Diagram
            container_diagram = self.architecture_generator.generate_container_diagram()
            self._create_container_diagram_doc(architecture_dir, container_diagram)
            
            # Generate C4 Level 3: Component Diagram
            component_diagram = self.architecture_generator.generate_component_diagram()
            self._create_component_diagram_doc(architecture_dir, component_diagram)
            
            # Generate C4 Level 4: Code Diagram
            code_diagram = self.architecture_generator.generate_code_diagram()
            self._create_code_diagram_doc(architecture_dir, code_diagram)
            
            # Update architecture index
            self._create_architecture_index(architecture_dir)
            
            print("  ✅ Architecture documentation generated")
            
        except Exception as e:
            print(f"  ❌ Failed to generate architecture docs: {e}")
    
    def _create_system_context_doc(self, output_dir: Path, context_data: Dict[str, Any]):
        """Create system context documentation."""
        content = f"""# System Context Diagram (C4 Level 1)

## Overview

{context_data.get('description', 'System context and external dependencies')}

## System Context

```mermaid
{self.architecture_generator.c4_generator.generate_system_context_diagram(context_data)}```

{self.architecture_generator.templates.create_diagram_legend('system_context')}

## Diagram Validation

"""
        
        # Validate the generated diagram
        diagram_content = self.architecture_generator.c4_generator.generate_system_context_diagram(context_data)
        validation = self.architecture_generator.templates.validate_diagram_syntax(diagram_content)
        
        if not validation['valid']:
            content += "⚠️ **Diagram Validation Issues:**\n"
            for error in validation['errors']:
                content += f"- Error: {error}\n"
        
        if validation['warnings']:
            for warning in validation['warnings']:
                content += f"- Warning: {warning}\n"
        
        content += "\n"
        
        # Add detailed descriptions
        content += "## System Users\n\n"
        for user in context_data.get('users', []):
            content += f"### {user}\n"
            content += f"Primary user type interacting with the {context_data['system_name']}.\n\n"
        
        content += "## External Systems\n\n"
        for ext_system in context_data.get('external_systems', []):
            content += f"### {ext_system}\n"
            content += f"External dependency providing services to the main system.\n\n"
        
        with open(output_dir / "01-system-context.md", 'w') as f:
            f.write(content)
    
    def _create_container_diagram_doc(self, output_dir: Path, container_data: Dict[str, Any]):
        """Create container diagram documentation."""
        content = """# Container Diagram (C4 Level 2)

## Overview

This diagram shows the high-level technology choices and how responsibilities are distributed across containers.

## Container Architecture

```mermaid
"""
        
        # Generate container diagram using template
        content += self.architecture_generator.templates.create_container_template("Container Diagram - System Architecture")
        
        # Add containers
        for i, container in enumerate(container_data.get('containers', [])):
            container_id = f"container_{i}"
            content += self.architecture_generator.templates.format_container(
                container_id, 
                container["name"], 
                container["technology"], 
                container["description"]
            ) + '\n'
        
        # Add basic relationships between containers
        containers = container_data.get('containers', [])
        if len(containers) > 1:
            content += "\n"
            # Frontend -> Backend relationship
            frontend_containers = [i for i, c in enumerate(containers) if 'web' in c['name'].lower() or 'frontend' in c['name'].lower()]
            backend_containers = [i for i, c in enumerate(containers) if 'api' in c['name'].lower() or 'backend' in c['name'].lower()]
            
            for fe_idx in frontend_containers:
                for be_idx in backend_containers:
                    content += self.architecture_generator.templates.format_relationship(f'container_{fe_idx}', f'container_{be_idx}', "Makes API calls", "HTTPS/REST") + '\n'
            
            # Backend -> Database relationship
            db_containers = [i for i, c in enumerate(containers) if 'database' in c['name'].lower()]
            for be_idx in backend_containers:
                for db_idx in db_containers:
                    content += self.architecture_generator.templates.format_relationship(f'container_{be_idx}', f'container_{db_idx}', "Reads/Writes", "SQL") + '\n'
        
        content += "```\n\n"
        content += self.architecture_generator.templates.create_diagram_legend('container')
        
        # Add container details
        content += "## Container Details\n\n"
        for container in containers:
            content += f"### {container['name']}\n\n"
            content += f"**Technology**: {container['technology']}\n\n"
            content += f"**Description**: {container['description']}\n\n"
            
            if 'responsibilities' in container:
                content += "**Key Responsibilities**:\n"
                for responsibility in container['responsibilities']:
                    content += f"- {responsibility}\n"
                content += "\n"
        
        with open(output_dir / "02-container-diagram.md", 'w') as f:
            f.write(content)
    
    def _create_component_diagram_doc(self, output_dir: Path, component_data: Dict[str, Any]):
        """Create component diagram documentation."""
        content = """# Component Diagram (C4 Level 3)

## Overview

This diagram shows the internal structure of containers, focusing on the backend components and their relationships.

## Component Architecture

```mermaid
"""
        
        # Generate component diagram using template
        content += self.architecture_generator.templates.create_component_template("Component Diagram - Backend Components")
        
        # Add components with proper indentation for container boundary
        for i, component in enumerate(component_data.get('components', [])):
            comp_id = f"comp_{i}"
            comp_type = component.get('type', 'component')
            technology = component.get('technology', 'Python')
            
            content += self.architecture_generator.templates.format_component(
                comp_id, 
                component["name"], 
                technology, 
                component["description"]
            ) + '\n'
        
        content += "    }\n\n"
        
        # Add relationships
        relationships = component_data.get('relationships', [])
        if relationships:
            content += "\n"
            # Create a mapping of component names to IDs
            comp_name_to_id = {comp['name']: f"comp_{i}" for i, comp in enumerate(component_data.get('components', []))}
            
            for rel in relationships:
                from_id = comp_name_to_id.get(rel['from'])
                to_id = comp_name_to_id.get(rel['to'])
                if from_id and to_id:
                    rel_type = rel.get('type', 'uses')
                    description = rel.get('description', f"{rel_type} relationship")
                    content += self.architecture_generator.templates.format_relationship(from_id, to_id, rel_type, description) + '\n'
        
        content += "```\n\n"
        content += self.architecture_generator.templates.create_diagram_legend('component')
        
        # Add component details
        content += "## Component Details\n\n"
        for component in component_data.get('components', []):
            content += f"### {component['name']}\n\n"
            content += f"**Type**: {component.get('type', 'component').title()}\n\n"
            content += f"**Technology**: {component.get('technology', 'Python')}\n\n"
            content += f"**Description**: {component['description']}\n\n"
            
            if 'responsibilities' in component:
                content += "**Responsibilities**:\n"
                for responsibility in component['responsibilities']:
                    content += f"- {responsibility}\n"
                content += "\n"
            
            if 'interfaces' in component:
                content += "**Interfaces**:\n"
                for interface in component['interfaces']:
                    content += f"- {interface}\n"
                content += "\n"
            
            if 'file' in component:
                content += f"**Source**: `{component['file']}`\n\n"
        
        with open(output_dir / "03-component-diagram.md", 'w') as f:
            f.write(content)
    
    def _create_code_diagram_doc(self, output_dir: Path, code_data: Dict[str, Any]):
        """Create code diagram documentation."""
        content = """# Code Diagram (C4 Level 4)

## Overview

This diagram shows the detailed class structure and relationships within the codebase.

## Class Structure

```mermaid
"""
        
        # Generate class diagram using template
        content += self.architecture_generator.templates.create_class_diagram_template()
        
        # Add classes using template
        classes = code_data.get('classes', [])
        for class_info in classes[:20]:  # Limit to first 20 classes for readability
            class_name = class_info['name']
            
            # Extract attributes and methods
            attributes = [attr['name'] for attr in class_info.get('attributes', [])]
            methods = [method['name'] for method in class_info.get('methods', [])]
            
            content += self.architecture_generator.templates.format_class_definition(class_name, attributes, methods)
        
        # Add relationships using template
        relationships = code_data.get('relationships', [])
        class_names = {class_info['name'] for class_info in classes[:20]}
        
        for rel in relationships:
            # Only include relationships between classes we're showing
            if rel['from'] in class_names and rel['to'] in class_names:
                rel_type = rel.get('type', 'uses')
                content += self.architecture_generator.templates.format_class_relationship(rel['from'], rel['to'], rel_type) + '\n'
        
        content += "```\n\n"
        content += self.architecture_generator.templates.create_diagram_legend('code')
        
        # Add class details
        content += "## Class Details\n\n"
        for class_info in classes:
            content += f"### {class_info['name']}\n\n"
            content += f"**File**: `{class_info['file']}`\n\n"
            
            if class_info.get('docstring'):
                content += f"**Description**: {class_info['docstring'].split('.')[0]}.\n\n"
            
            if class_info.get('base_classes'):
                content += "**Inherits from**: " + ", ".join(class_info['base_classes']) + "\n\n"
            
            # Add method details
            methods = class_info.get('methods', [])
            if methods:
                content += "**Methods**:\n\n"
                for method in methods:
                    content += f"- `{method['name']}()`: "
                    if method.get('docstring'):
                        content += method['docstring'].split('.')[0] + "."
                    else:
                        content += f"{method['name'].replace('_', ' ').title()} functionality."
                    content += "\n"
                content += "\n"
        
        # Add interfaces
        interfaces = code_data.get('interfaces', [])
        if interfaces:
            content += "## Interfaces\n\n"
            for interface in interfaces:
                content += f"### {interface['name']}\n\n"
                content += f"**Type**: {interface['type'].title()}\n\n"
                content += f"**Description**: {interface['description']}\n\n"
                
                if interface.get('methods'):
                    content += "**Methods**:\n"
                    for method in interface['methods']:
                        content += f"- `{method}`\n"
                    content += "\n"
        
        with open(output_dir / "04-code-diagram.md", 'w') as f:
            f.write(content)
    
    def _create_architecture_index(self, output_dir: Path):
        """Create architecture documentation index."""
        content = f"""# Architecture Documentation

This section provides comprehensive architectural documentation following the C4 Model methodology.

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## C4 Model Hierarchy

The C4 model provides a hierarchical approach to documenting software architecture:

### Level 1: System Context
- [System Context Diagram](./01-system-context.md) - Shows the system in its environment

### Level 2: Container Architecture  
- [Container Diagram](./02-container-diagram.md) - Shows high-level technology choices

### Level 3: Component Architecture
- [Component Diagram](./03-component-diagram.md) - Shows internal component structure

### Level 4: Code Architecture
- [Code Diagram](./04-code-diagram.md) - Shows detailed class relationships

## Architecture Principles

This system follows these key architectural principles:

- **Layered Architecture**: Clear separation between API, service, and data layers
- **Dependency Injection**: Loose coupling through dependency injection patterns
- **Single Responsibility**: Each component has a focused, well-defined purpose
- **Interface Segregation**: Components depend on abstractions, not concretions

## Technology Stack

The system is built using:

- **Backend**: Python with FastAPI framework
- **Frontend**: React with TypeScript
- **Database**: SQLite with SQLAlchemy ORM
- **ML Framework**: PyTorch with Vision Transformers
- **Deployment**: Docker containerization

## Quality Attributes

Key quality attributes addressed by this architecture:

- **Maintainability**: Modular design with clear boundaries
- **Testability**: Dependency injection enables comprehensive testing
- **Scalability**: Stateless services support horizontal scaling
- **Performance**: Optimized ML pipeline with caching strategies
"""
        
        with open(output_dir / "README.md", 'w') as f:
            f.write(content)
    
    def update_documentation_index(self):
        """Update the main documentation index with generated content."""
        print("📚 Updating documentation index...")
        
        # Update main README with current status
        readme_content = f"""# Children's Drawing Anomaly Detection System - Documentation

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Version**: 1.0.0  
**Status**: Auto-generated from codebase

## Quick Navigation

### 🏗️ Architecture
- [System Overview](./architecture/01-system-context.md)
- [Component Architecture](./architecture/06-backend-components.md)
- [Database Schema](./database/schema.md)

### 📡 API Reference
- [API Overview](./api/README.md)
- [OpenAPI Schema](./api/openapi.json)
- [Endpoint Documentation](./api/endpoints/)

### 🧮 Algorithms
- [Score Normalization](./algorithms/07-score-normalization.md)
- [Algorithm Implementations](./algorithms/implementations/)

### 🔄 Workflows
- [Upload and Analysis](./workflows/business/01-drawing-upload-analysis.md)
- [System Workflows](./workflows/README.md)

### 🔌 Interfaces
- [Service Interfaces](./interfaces/services/)
- [API Contracts](./interfaces/api/)

### 🚀 Deployment
- [Docker Deployment](./deployment/docker.md)
- [Environment Setup](./deployment/environment-setup.md)

## Documentation Standards

This documentation follows industry standards:
- **C4 Model** for architecture
- **OpenAPI 3.0** for API documentation
- **BPMN 2.0** for process flows
- **UML 2.5** for interfaces
- **IEEE 830** for requirements

## Maintenance

Documentation is automatically generated from code using:
```bash
python scripts/generate_docs.py
```

For manual updates, see [Documentation Guidelines](./CONTRIBUTING.md).
"""
        
        with open(self.docs_dir / "README.md", 'w') as f:
            f.write(readme_content)
        
        print("  ✅ Documentation index updated")

def main():
    """Main entry point for documentation generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate comprehensive documentation")
    parser.add_argument("--force", action="store_true", 
                       help="Force regeneration of all documentation")
    parser.add_argument("--category", choices=[t.value for t in DocumentationType],
                       help="Generate documentation for specific category only")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate existing documentation")
    parser.add_argument("--clear-cache", action="store_true",
                       help="Clear documentation cache before generation")
    parser.add_argument("--cache-status", action="store_true",
                       help="Show cache status and exit")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    engine = DocumentationEngine(project_root)
    
    if args.cache_status:
        print("📊 Documentation Cache Status:")
        status = engine.get_cache_status()
        for key, value in status.items():
            print(f"  - {key.replace('_', ' ').title()}: {value}")
        return
    
    if args.clear_cache:
        engine.clear_cache()
    
    if args.validate_only:
        print("🔍 Validating documentation...")
        result = engine.validate_sources()
        if result.is_valid:
            print("✅ All documentation is valid")
        else:
            print("❌ Documentation validation failed:")
            for error in result.errors:
                print(f"  - {error}")
        return
    
    if args.category:
        print(f"📝 Generating {args.category} documentation...")
        doc_type = DocumentationType(args.category)
        result = engine.generate_category(doc_type)
    else:
        result = engine.generate_all(force=args.force)
    
    # Print summary
    if result.success:
        print(f"\n📊 Generation Summary:")
        print(f"  - Files generated: {len(result.generated_files)}")
        print(f"  - Changes detected: {len(result.changes_detected)}")
        print(f"  - Duration: {result.duration:.2f}s")
        
        if result.warnings:
            print(f"  - Warnings: {len(result.warnings)}")
            for warning in result.warnings[:5]:  # Show first 5 warnings
                print(f"    ⚠️  {warning}")
    else:
        print(f"\n❌ Generation failed:")
        for error in result.errors:
            print(f"  - {error}")


# Maintain backward compatibility
DocumentationGenerator = DocumentationEngine

if __name__ == "__main__":
    main()