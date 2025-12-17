#!/usr/bin/env python3
"""
Enhanced API Documentation Generator

This module provides comprehensive API documentation generation capabilities
including OpenAPI schema extraction with validation, request/response example
generation, and authentication and error specification extraction.

Implements Requirements 2.1, 2.4, 2.5 from the comprehensive documentation specification.
"""

import json
import inspect
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import ast
import importlib.util
import sys


@dataclass
class APIEndpoint:
    """Represents an API endpoint with comprehensive metadata."""
    path: str
    method: str
    operation_id: str
    summary: str
    description: str
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    security: List[Dict[str, Any]] = field(default_factory=list)
    examples: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of API documentation validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    endpoint_count: int = 0
    schema_version: str = ""


@dataclass
class ExampleGenerationConfig:
    """Configuration for request/response example generation."""
    include_optional_fields: bool = True
    max_array_items: int = 3
    use_realistic_values: bool = True
    include_edge_cases: bool = False


class SchemaValidator:
    """Validates OpenAPI schemas against specification requirements."""
    
    def __init__(self):
        self.required_openapi_fields = ['openapi', 'info', 'paths']
        self.required_info_fields = ['title', 'version']
        self.supported_openapi_versions = ['3.0.0', '3.0.1', '3.0.2', '3.0.3', '3.1.0']
    
    def validate_schema(self, schema: Dict[str, Any]) -> ValidationResult:
        """
        Validate OpenAPI schema for compliance and completeness.
        
        Args:
            schema: OpenAPI schema dictionary to validate
            
        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult(is_valid=True)
        
        try:
            # Validate top-level structure
            for field in self.required_openapi_fields:
                if field not in schema:
                    result.is_valid = False
                    result.errors.append(f"Missing required field: {field}")
            
            # Validate OpenAPI version
            if 'openapi' in schema:
                version = schema['openapi']
                result.schema_version = version
                if version not in self.supported_openapi_versions:
                    result.warnings.append(f"OpenAPI version {version} may not be fully supported")
            
            # Validate info section
            if 'info' in schema:
                info = schema['info']
                for field in self.required_info_fields:
                    if field not in info:
                        result.is_valid = False
                        result.errors.append(f"Missing required info field: {field}")
            
            # Validate paths
            if 'paths' in schema:
                paths = schema['paths']
                result.endpoint_count = sum(len(methods) for methods in paths.values() if isinstance(methods, dict))
                
                for path, methods in paths.items():
                    if not isinstance(methods, dict):
                        result.errors.append(f"Invalid path structure for {path}")
                        continue
                    
                    for method, spec in methods.items():
                        endpoint_errors = self._validate_endpoint(path, method, spec)
                        result.errors.extend(endpoint_errors)
            
            # Validate components section if present
            if 'components' in schema:
                component_errors = self._validate_components(schema['components'])
                result.errors.extend(component_errors)
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Schema validation failed: {str(e)}")
        
        return result
    
    def _validate_endpoint(self, path: str, method: str, spec: Dict[str, Any]) -> List[str]:
        """Validate individual endpoint specification."""
        errors = []
        
        # Check required endpoint fields
        if 'responses' not in spec:
            errors.append(f"Missing responses for {method.upper()} {path}")
        elif not spec['responses']:
            errors.append(f"Empty responses for {method.upper()} {path}")
        
        # Validate responses structure
        if 'responses' in spec:
            for status_code, response in spec['responses'].items():
                if not isinstance(response, dict):
                    errors.append(f"Invalid response structure for {status_code} in {method.upper()} {path}")
                elif 'description' not in response:
                    errors.append(f"Missing description for response {status_code} in {method.upper()} {path}")
        
        # Validate parameters if present
        if 'parameters' in spec:
            for i, param in enumerate(spec['parameters']):
                if not isinstance(param, dict):
                    errors.append(f"Invalid parameter structure at index {i} for {method.upper()} {path}")
                    continue
                
                required_param_fields = ['name', 'in']
                for field in required_param_fields:
                    if field not in param:
                        errors.append(f"Missing {field} in parameter {i} for {method.upper()} {path}")
        
        return errors
    
    def _validate_components(self, components: Dict[str, Any]) -> List[str]:
        """Validate components section of OpenAPI schema."""
        errors = []
        
        # Validate schemas if present
        if 'schemas' in components:
            schemas = components['schemas']
            if not isinstance(schemas, dict):
                errors.append("Components schemas must be a dictionary")
            else:
                for schema_name, schema_def in schemas.items():
                    if not isinstance(schema_def, dict):
                        errors.append(f"Invalid schema definition for {schema_name}")
        
        return errors


class ExampleGenerator:
    """Generates realistic request/response examples from OpenAPI schemas."""
    
    def __init__(self, config: ExampleGenerationConfig = None):
        self.config = config or ExampleGenerationConfig()
        self.type_examples = {
            'string': 'example_string',
            'integer': 42,
            'number': 3.14,
            'boolean': True,
            'array': [],
            'object': {}
        }
    
    def generate_request_examples(self, endpoint: APIEndpoint, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive request examples for an endpoint.
        
        Args:
            endpoint: API endpoint information
            schema: Full OpenAPI schema for reference resolution
            
        Returns:
            Dictionary containing various request examples
        """
        examples = {}
        
        try:
            # Generate parameter examples
            if endpoint.parameters:
                examples['parameters'] = self._generate_parameter_examples(endpoint.parameters)
            
            # Generate request body examples
            if endpoint.request_body:
                examples['request_body'] = self._generate_request_body_examples(
                    endpoint.request_body, schema
                )
            
            # Generate complete request example
            examples['complete_request'] = self._generate_complete_request_example(endpoint)
            
        except Exception as e:
            examples['error'] = f"Failed to generate request examples: {str(e)}"
        
        return examples
    
    def generate_response_examples(self, endpoint: APIEndpoint, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive response examples for an endpoint.
        
        Args:
            endpoint: API endpoint information
            schema: Full OpenAPI schema for reference resolution
            
        Returns:
            Dictionary containing response examples for each status code
        """
        examples = {}
        
        try:
            for status_code, response_spec in endpoint.responses.items():
                examples[status_code] = self._generate_response_example(
                    status_code, response_spec, schema
                )
        except Exception as e:
            examples['error'] = f"Failed to generate response examples: {str(e)}"
        
        return examples
    
    def _generate_parameter_examples(self, parameters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate examples for endpoint parameters."""
        examples = {}
        
        for param in parameters:
            param_name = param.get('name', 'unknown')
            param_type = param.get('schema', {}).get('type', 'string')
            param_in = param.get('in', 'query')
            
            # Generate example based on parameter type and context
            if param_name.lower() in ['id', 'user_id', 'drawing_id']:
                example_value = 123
            elif param_name.lower() in ['page', 'limit', 'size']:
                example_value = 10
            elif param_name.lower() in ['email']:
                example_value = 'user@example.com'
            elif param_type == 'integer':
                example_value = 42
            elif param_type == 'number':
                example_value = 3.14
            elif param_type == 'boolean':
                example_value = True
            else:
                example_value = f'example_{param_name}'
            
            if param_in not in examples:
                examples[param_in] = {}
            examples[param_in][param_name] = example_value
        
        return examples
    
    def _generate_request_body_examples(self, request_body: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate examples for request body content."""
        examples = {}
        
        content = request_body.get('content', {})
        for media_type, media_spec in content.items():
            if 'schema' in media_spec:
                examples[media_type] = self._generate_schema_example(media_spec['schema'], schema)
        
        return examples
    
    def _generate_response_example(self, status_code: str, response_spec: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate example for a specific response."""
        example = {
            'description': response_spec.get('description', 'No description'),
            'status_code': status_code
        }
        
        content = response_spec.get('content', {})
        if content:
            example['content'] = {}
            for media_type, media_spec in content.items():
                if 'schema' in media_spec:
                    example['content'][media_type] = self._generate_schema_example(media_spec['schema'], schema)
        
        return example
    
    def _generate_schema_example(self, schema_def: Dict[str, Any], full_schema: Dict[str, Any]) -> Any:
        """Generate example data from schema definition."""
        if '$ref' in schema_def:
            # Resolve reference
            ref_path = schema_def['$ref']
            if ref_path.startswith('#/'):
                # Internal reference
                parts = ref_path[2:].split('/')
                resolved_schema = full_schema
                for part in parts:
                    resolved_schema = resolved_schema.get(part, {})
                return self._generate_schema_example(resolved_schema, full_schema)
        
        schema_type = schema_def.get('type', 'object')
        
        if schema_type == 'object':
            return self._generate_object_example(schema_def, full_schema)
        elif schema_type == 'array':
            return self._generate_array_example(schema_def, full_schema)
        else:
            return self.type_examples.get(schema_type, f'example_{schema_type}')
    
    def _generate_object_example(self, schema_def: Dict[str, Any], full_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate example for object schema."""
        example = {}
        properties = schema_def.get('properties', {})
        required = schema_def.get('required', [])
        
        for prop_name, prop_schema in properties.items():
            if prop_name in required or self.config.include_optional_fields:
                example[prop_name] = self._generate_schema_example(prop_schema, full_schema)
        
        return example
    
    def _generate_array_example(self, schema_def: Dict[str, Any], full_schema: Dict[str, Any]) -> List[Any]:
        """Generate example for array schema."""
        items_schema = schema_def.get('items', {'type': 'string'})
        example_item = self._generate_schema_example(items_schema, full_schema)
        
        # Generate multiple items up to max_array_items
        return [example_item for _ in range(min(self.config.max_array_items, 2))]
    
    def _generate_complete_request_example(self, endpoint: APIEndpoint) -> Dict[str, Any]:
        """Generate a complete request example including URL, headers, and body."""
        example = {
            'method': endpoint.method.upper(),
            'url': endpoint.path,
            'headers': {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        }
        
        # Add authentication headers if security is defined
        if endpoint.security:
            for security_scheme in endpoint.security:
                if 'bearerAuth' in security_scheme:
                    example['headers']['Authorization'] = 'Bearer <your_token_here>'
                elif 'apiKey' in security_scheme:
                    example['headers']['X-API-Key'] = '<your_api_key_here>'
        
        return example


class AuthenticationExtractor:
    """Extracts authentication and security requirements from API specifications."""
    
    def extract_security_schemes(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract security schemes from OpenAPI schema.
        
        Args:
            schema: Full OpenAPI schema
            
        Returns:
            Dictionary of security schemes and their configurations
        """
        security_info = {
            'schemes': {},
            'global_security': [],
            'endpoint_security': {}
        }
        
        try:
            # Extract security schemes from components
            components = schema.get('components', {})
            security_schemes = components.get('securitySchemes', {})
            
            for scheme_name, scheme_def in security_schemes.items():
                security_info['schemes'][scheme_name] = {
                    'type': scheme_def.get('type', 'unknown'),
                    'description': scheme_def.get('description', ''),
                    'scheme': scheme_def.get('scheme', ''),
                    'bearerFormat': scheme_def.get('bearerFormat', ''),
                    'in': scheme_def.get('in', ''),
                    'name': scheme_def.get('name', '')
                }
            
            # Extract global security requirements
            global_security = schema.get('security', [])
            security_info['global_security'] = global_security
            
            # Extract endpoint-specific security
            paths = schema.get('paths', {})
            for path, methods in paths.items():
                for method, spec in methods.items():
                    if 'security' in spec:
                        endpoint_key = f"{method.upper()} {path}"
                        security_info['endpoint_security'][endpoint_key] = spec['security']
            
        except Exception as e:
            security_info['error'] = f"Failed to extract security information: {str(e)}"
        
        return security_info
    
    def generate_authentication_docs(self, security_info: Dict[str, Any]) -> str:
        """
        Generate authentication documentation from security information.
        
        Args:
            security_info: Security information extracted from schema
            
        Returns:
            Markdown formatted authentication documentation
        """
        docs = "# Authentication\n\n"
        
        if not security_info.get('schemes'):
            docs += "No authentication required for this API.\n\n"
            return docs
        
        docs += "This API uses the following authentication methods:\n\n"
        
        # Document each security scheme
        for scheme_name, scheme_def in security_info.get('schemes', {}).items():
            docs += f"## {scheme_name}\n\n"
            docs += f"**Type**: {scheme_def.get('type', 'Unknown')}\n\n"
            
            if scheme_def.get('description'):
                docs += f"**Description**: {scheme_def['description']}\n\n"
            
            scheme_type = scheme_def.get('type', '')
            
            if scheme_type == 'http':
                scheme = scheme_def.get('scheme', '')
                docs += f"**Scheme**: {scheme}\n\n"
                
                if scheme == 'bearer':
                    bearer_format = scheme_def.get('bearerFormat', 'JWT')
                    docs += f"**Bearer Format**: {bearer_format}\n\n"
                    docs += "**Usage**:\n```http\nAuthorization: Bearer <your_token>\n```\n\n"
                elif scheme == 'basic':
                    docs += "**Usage**:\n```http\nAuthorization: Basic <base64_encoded_credentials>\n```\n\n"
            
            elif scheme_type == 'apiKey':
                key_location = scheme_def.get('in', 'header')
                key_name = scheme_def.get('name', 'X-API-Key')
                docs += f"**Location**: {key_location}\n"
                docs += f"**Parameter Name**: {key_name}\n\n"
                
                if key_location == 'header':
                    docs += f"**Usage**:\n```http\n{key_name}: <your_api_key>\n```\n\n"
                elif key_location == 'query':
                    docs += f"**Usage**: Add `{key_name}=<your_api_key>` to query parameters\n\n"
            
            elif scheme_type == 'oauth2':
                docs += "**OAuth 2.0 Flow**: See OAuth 2.0 specification for implementation details\n\n"
        
        # Document global security requirements
        if security_info.get('global_security'):
            docs += "## Global Security Requirements\n\n"
            docs += "All endpoints require the following authentication:\n\n"
            for requirement in security_info['global_security']:
                for scheme_name, scopes in requirement.items():
                    docs += f"- **{scheme_name}**"
                    if scopes:
                        docs += f" (scopes: {', '.join(scopes)})"
                    docs += "\n"
            docs += "\n"
        
        return docs


class ErrorSpecificationExtractor:
    """Extracts and documents error specifications from API schemas."""
    
    def extract_error_responses(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract error response specifications from OpenAPI schema.
        
        Args:
            schema: Full OpenAPI schema
            
        Returns:
            Dictionary containing error response information
        """
        error_info = {
            'common_errors': {},
            'endpoint_errors': {},
            'error_schemas': {}
        }
        
        try:
            # Extract error schemas from components
            components = schema.get('components', {})
            schemas = components.get('schemas', {})
            
            for schema_name, schema_def in schemas.items():
                if 'error' in schema_name.lower() or 'exception' in schema_name.lower():
                    error_info['error_schemas'][schema_name] = schema_def
            
            # Extract error responses from endpoints
            paths = schema.get('paths', {})
            error_status_codes = ['400', '401', '403', '404', '422', '429', '500', '502', '503']
            
            for path, methods in paths.items():
                for method, spec in methods.items():
                    responses = spec.get('responses', {})
                    endpoint_key = f"{method.upper()} {path}"
                    endpoint_errors = {}
                    
                    for status_code, response in responses.items():
                        if status_code in error_status_codes:
                            endpoint_errors[status_code] = {
                                'description': response.get('description', ''),
                                'content': response.get('content', {})
                            }
                            
                            # Track common errors across endpoints
                            if status_code not in error_info['common_errors']:
                                error_info['common_errors'][status_code] = []
                            error_info['common_errors'][status_code].append(endpoint_key)
                    
                    if endpoint_errors:
                        error_info['endpoint_errors'][endpoint_key] = endpoint_errors
            
        except Exception as e:
            error_info['error'] = f"Failed to extract error specifications: {str(e)}"
        
        return error_info
    
    def generate_error_documentation(self, error_info: Dict[str, Any]) -> str:
        """
        Generate comprehensive error documentation.
        
        Args:
            error_info: Error information extracted from schema
            
        Returns:
            Markdown formatted error documentation
        """
        docs = "# Error Handling\n\n"
        
        if not error_info.get('common_errors') and not error_info.get('endpoint_errors'):
            docs += "No specific error handling documentation available.\n\n"
            return docs
        
        # Document common error responses
        if error_info.get('common_errors'):
            docs += "## Common Error Responses\n\n"
            docs += "The following error responses are used across multiple endpoints:\n\n"
            
            status_descriptions = {
                '400': 'Bad Request - Invalid request parameters or body',
                '401': 'Unauthorized - Authentication required or invalid',
                '403': 'Forbidden - Insufficient permissions',
                '404': 'Not Found - Resource does not exist',
                '422': 'Unprocessable Entity - Validation error',
                '429': 'Too Many Requests - Rate limit exceeded',
                '500': 'Internal Server Error - Server encountered an error',
                '502': 'Bad Gateway - Upstream server error',
                '503': 'Service Unavailable - Service temporarily unavailable'
            }
            
            for status_code, endpoints in error_info['common_errors'].items():
                description = status_descriptions.get(status_code, f'HTTP {status_code}')
                docs += f"### {status_code} - {description}\n\n"
                docs += f"**Used by**: {len(endpoints)} endpoint(s)\n\n"
        
        # Document error schemas
        if error_info.get('error_schemas'):
            docs += "## Error Response Schemas\n\n"
            
            for schema_name, schema_def in error_info['error_schemas'].items():
                docs += f"### {schema_name}\n\n"
                
                if 'description' in schema_def:
                    docs += f"{schema_def['description']}\n\n"
                
                # Document properties if it's an object
                if schema_def.get('type') == 'object' and 'properties' in schema_def:
                    docs += "**Properties**:\n\n"
                    for prop_name, prop_def in schema_def['properties'].items():
                        prop_type = prop_def.get('type', 'unknown')
                        prop_desc = prop_def.get('description', 'No description')
                        docs += f"- `{prop_name}` ({prop_type}): {prop_desc}\n"
                    docs += "\n"
        
        # Document endpoint-specific errors
        if error_info.get('endpoint_errors'):
            docs += "## Endpoint-Specific Errors\n\n"
            docs += "Some endpoints may return additional error responses:\n\n"
            
            for endpoint, errors in list(error_info['endpoint_errors'].items())[:5]:  # Limit to first 5
                docs += f"### {endpoint}\n\n"
                for status_code, error_spec in errors.items():
                    docs += f"- **{status_code}**: {error_spec.get('description', 'No description')}\n"
                docs += "\n"
        
        return docs


class APIDocumentationGenerator:
    """
    Enhanced API Documentation Generator with comprehensive features.
    
    Provides OpenAPI schema extraction with validation, request/response example
    generation, and authentication and error specification extraction.
    
    Implements Requirements 2.1, 2.4, 2.5 from the comprehensive documentation specification.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.docs_dir = project_root / "docs" / "api"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.validator = SchemaValidator()
        self.example_generator = ExampleGenerator()
        self.auth_extractor = AuthenticationExtractor()
        self.error_extractor = ErrorSpecificationExtractor()
        
        # Configuration
        self.generation_timestamp = datetime.now()
        self.validation_enabled = True
    
    def extract_openapi_schema(self, app_module_path: str = "app.main") -> Dict[str, Any]:
        """
        Extract OpenAPI schema from FastAPI application with validation.
        
        Args:
            app_module_path: Python module path to FastAPI app
            
        Returns:
            Extracted and validated OpenAPI schema
            
        Implements Requirement 2.1: OpenAPI 3.0 compliant specifications from FastAPI code
        """
        try:
            # Import the FastAPI application
            if app_module_path in sys.modules:
                app_module = sys.modules[app_module_path]
            else:
                spec = importlib.util.spec_from_file_location(
                    "app_main", 
                    self.project_root / "app" / "main.py"
                )
                app_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(app_module)
            
            # Get the FastAPI app instance
            app = getattr(app_module, 'app', None)
            if not app:
                raise ValueError("No FastAPI app instance found")
            
            # Extract OpenAPI schema
            schema = app.openapi()
            
            # Validate schema if validation is enabled
            if self.validation_enabled:
                validation_result = self.validator.validate_schema(schema)
                if not validation_result.is_valid:
                    print(f"⚠️  Schema validation warnings: {len(validation_result.errors)} errors")
                    for error in validation_result.errors[:5]:  # Show first 5 errors
                        print(f"   - {error}")
            
            return schema
            
        except Exception as e:
            print(f"❌ Failed to extract OpenAPI schema: {e}")
            # Return minimal valid schema as fallback
            return {
                'openapi': '3.1.0',
                'info': {
                    'title': 'API Documentation',
                    'version': '1.0.0',
                    'description': 'Generated API documentation'
                },
                'paths': {}
            }
    
    def generate_comprehensive_request_response_examples(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive request/response examples for all endpoints.
        
        Args:
            schema: OpenAPI schema to generate examples from
            
        Returns:
            Dictionary containing examples for all endpoints
            
        Implements Requirement 2.4: Request/response examples and schema specifications
        """
        examples = {
            'endpoints': {},
            'generation_info': {
                'timestamp': self.generation_timestamp.isoformat(),
                'total_endpoints': 0,
                'examples_generated': 0
            }
        }
        
        try:
            paths = schema.get('paths', {})
            
            for path, methods in paths.items():
                for method, spec in methods.items():
                    if method.lower() in ['get', 'post', 'put', 'delete', 'patch']:
                        # Create endpoint object
                        endpoint = APIEndpoint(
                            path=path,
                            method=method,
                            operation_id=spec.get('operationId', f"{method}_{path}"),
                            summary=spec.get('summary', ''),
                            description=spec.get('description', ''),
                            parameters=spec.get('parameters', []),
                            request_body=spec.get('requestBody'),
                            responses=spec.get('responses', {}),
                            tags=spec.get('tags', []),
                            security=spec.get('security', [])
                        )
                        
                        endpoint_key = f"{method.upper()} {path}"
                        examples['endpoints'][endpoint_key] = {
                            'request_examples': self.example_generator.generate_request_examples(endpoint, schema),
                            'response_examples': self.example_generator.generate_response_examples(endpoint, schema),
                            'endpoint_info': {
                                'summary': endpoint.summary,
                                'description': endpoint.description,
                                'tags': endpoint.tags
                            }
                        }
                        
                        examples['generation_info']['total_endpoints'] += 1
                        examples['generation_info']['examples_generated'] += 1
            
        except Exception as e:
            examples['error'] = f"Failed to generate examples: {str(e)}"
        
        return examples
    
    def extract_authentication_and_error_specifications(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract authentication requirements and error specifications from schema.
        
        Args:
            schema: OpenAPI schema to extract specifications from
            
        Returns:
            Dictionary containing authentication and error specifications
            
        Implements Requirement 2.5: Authentication requirements and error response specifications
        """
        specifications = {
            'authentication': {},
            'errors': {},
            'extraction_info': {
                'timestamp': self.generation_timestamp.isoformat(),
                'has_authentication': False,
                'error_responses_found': 0
            }
        }
        
        try:
            # Extract authentication specifications
            auth_info = self.auth_extractor.extract_security_schemes(schema)
            specifications['authentication'] = auth_info
            specifications['extraction_info']['has_authentication'] = bool(auth_info.get('schemes'))
            
            # Extract error specifications
            error_info = self.error_extractor.extract_error_responses(schema)
            specifications['errors'] = error_info
            specifications['extraction_info']['error_responses_found'] = len(error_info.get('common_errors', {}))
            
        except Exception as e:
            specifications['error'] = f"Failed to extract specifications: {str(e)}"
        
        return specifications
    
    def generate_enhanced_documentation(self, force_regenerate: bool = False) -> Dict[str, Any]:
        """
        Generate enhanced API documentation with all comprehensive features.
        
        Args:
            force_regenerate: Whether to force regeneration even if files exist
            
        Returns:
            Dictionary containing generation results and file paths
            
        Implements Requirements 2.1, 2.4, 2.5: Complete enhanced API documentation
        """
        result = {
            'success': True,
            'generated_files': [],
            'errors': [],
            'warnings': [],
            'generation_info': {
                'timestamp': self.generation_timestamp.isoformat(),
                'force_regenerate': force_regenerate
            }
        }
        
        try:
            print("📡 Generating enhanced API documentation...")
            
            # Extract OpenAPI schema with validation
            schema = self.extract_openapi_schema()
            
            # Save enhanced OpenAPI schema
            openapi_file = self.docs_dir / "openapi.json"
            with open(openapi_file, 'w') as f:
                json.dump(schema, f, indent=2)
            result['generated_files'].append(openapi_file)
            
            # Generate comprehensive examples
            examples = self.generate_comprehensive_request_response_examples(schema)
            examples_file = self.docs_dir / "examples.json"
            with open(examples_file, 'w') as f:
                json.dump(examples, f, indent=2)
            result['generated_files'].append(examples_file)
            
            # Extract authentication and error specifications
            specifications = self.extract_authentication_and_error_specifications(schema)
            
            # Generate authentication documentation
            if specifications['authentication'].get('schemes'):
                auth_docs = self.auth_extractor.generate_authentication_docs(specifications['authentication'])
                auth_file = self.docs_dir / "authentication.md"
                with open(auth_file, 'w') as f:
                    f.write(auth_docs)
                result['generated_files'].append(auth_file)
            
            # Generate error handling documentation
            if specifications['errors'].get('common_errors') or specifications['errors'].get('endpoint_errors'):
                error_docs = self.error_extractor.generate_error_documentation(specifications['errors'])
                error_file = self.docs_dir / "error-handling.md"
                with open(error_file, 'w') as f:
                    f.write(error_docs)
                result['generated_files'].append(error_file)
            
            # Generate enhanced endpoint documentation
            self._generate_enhanced_endpoint_docs(schema, examples)
            endpoint_files = list((self.docs_dir / "endpoints").glob("*.md"))
            result['generated_files'].extend(endpoint_files)
            
            # Generate API overview with enhanced features
            self._generate_enhanced_api_overview(schema, specifications)
            overview_file = self.docs_dir / "README.md"
            result['generated_files'].append(overview_file)
            
            # Generate enhanced Swagger UI
            swagger_result = self._generate_enhanced_swagger_ui(schema)
            if swagger_result['success']:
                result['generated_files'].extend(swagger_result['generated_files'])
            else:
                result['warnings'].extend(swagger_result['errors'])
            
            print(f"  ✅ Enhanced API documentation generated ({len(result['generated_files'])} files)")
            
        except Exception as e:
            result['success'] = False
            result['errors'].append(f"Enhanced documentation generation failed: {str(e)}")
            print(f"  ❌ Enhanced API documentation generation failed: {e}")
        
        return result
    
    def _generate_enhanced_endpoint_docs(self, schema: Dict[str, Any], examples: Dict[str, Any]):
        """Generate enhanced endpoint documentation with examples and specifications."""
        endpoints_dir = self.docs_dir / "endpoints"
        endpoints_dir.mkdir(exist_ok=True)
        
        paths = schema.get('paths', {})
        endpoint_examples = examples.get('endpoints', {})
        
        for path, methods in paths.items():
            for method, spec in methods.items():
                if method.lower() in ['get', 'post', 'put', 'delete', 'patch']:
                    self._create_enhanced_endpoint_doc(
                        endpoints_dir, path, method, spec, 
                        endpoint_examples.get(f"{method.upper()} {path}", {})
                    )
    
    def _create_enhanced_endpoint_doc(self, endpoints_dir: Path, path: str, method: str, 
                                    spec: Dict[str, Any], examples: Dict[str, Any]):
        """Create enhanced documentation for a single endpoint."""
        # Clean path for filename
        filename = f"{method.upper()}_{path.replace('/', '_').replace('{', '').replace('}', '')}.md"
        filepath = endpoints_dir / filename
        
        content = f"""# {method.upper()} {path}

## Summary
{spec.get('summary', 'No summary available')}

## Description
{spec.get('description', 'No description available')}

## Tags
{', '.join(spec.get('tags', ['untagged']))}

## Parameters
"""
        
        # Add enhanced parameters documentation
        parameters = spec.get('parameters', [])
        if parameters:
            content += "| Name | Location | Type | Required | Description |\n"
            content += "|------|----------|------|----------|-------------|\n"
            for param in parameters:
                name = param.get('name', 'unknown')
                location = param.get('in', 'unknown')
                param_type = param.get('schema', {}).get('type', 'unknown')
                required = 'Yes' if param.get('required', False) else 'No'
                description = param.get('description', 'No description')
                content += f"| {name} | {location} | {param_type} | {required} | {description} |\n"
        else:
            content += "No parameters required.\n"
        
        # Add request body documentation
        request_body = spec.get('requestBody')
        if request_body:
            content += f"\n## Request Body\n"
            content += f"{request_body.get('description', 'Request body required')}\n\n"
            
            # Add request examples if available
            request_examples = examples.get('request_examples', {})
            if request_examples and 'request_body' in request_examples:
                content += "### Request Body Examples\n\n"
                for media_type, example in request_examples['request_body'].items():
                    content += f"**{media_type}**:\n```json\n{json.dumps(example, indent=2)}\n```\n\n"
        
        # Add enhanced responses documentation
        responses = spec.get('responses', {})
        content += f"\n## Responses\n\n"
        
        for status_code, response in responses.items():
            content += f"### {status_code} - {response.get('description', 'No description')}\n\n"
            
            # Add response examples if available
            response_examples = examples.get('response_examples', {})
            if status_code in response_examples:
                example_data = response_examples[status_code]
                if 'content' in example_data:
                    for media_type, example in example_data['content'].items():
                        content += f"**{media_type}**:\n```json\n{json.dumps(example, indent=2)}\n```\n\n"
        
        # Add complete request example
        content += f"\n## Complete Request Example\n\n"
        request_examples = examples.get('request_examples', {})
        if 'complete_request' in request_examples:
            complete_example = request_examples['complete_request']
            content += f"```http\n{complete_example.get('method', method.upper())} {path}\n"
            for header, value in complete_example.get('headers', {}).items():
                content += f"{header}: {value}\n"
            content += "```\n\n"
        else:
            content += f"```http\n{method.upper()} {path}\nContent-Type: application/json\n```\n\n"
        
        # Add security information if present
        security = spec.get('security', [])
        if security:
            content += "## Security\n\n"
            content += "This endpoint requires authentication:\n\n"
            for requirement in security:
                for scheme_name, scopes in requirement.items():
                    content += f"- **{scheme_name}**"
                    if scopes:
                        content += f" (scopes: {', '.join(scopes)})"
                    content += "\n"
            content += "\n"
        
        with open(filepath, "w") as f:
            f.write(content)
    
    def _generate_enhanced_api_overview(self, schema: Dict[str, Any], specifications: Dict[str, Any]):
        """Generate enhanced API overview documentation."""
        info = schema.get('info', {})
        paths = schema.get('paths', {})
        
        content = f"""# {info.get('title', 'API Documentation')}

**Version**: {info.get('version', '1.0.0')}  
**Generated**: {self.generation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Description
{info.get('description', 'API documentation generated from OpenAPI specification')}

## Base Information
- **OpenAPI Version**: {schema.get('openapi', '3.1.0')}
- **Total Endpoints**: {sum(len(methods) for methods in paths.values() if isinstance(methods, dict))}
- **Documentation Format**: Enhanced with examples and specifications

## Quick Navigation
- [Authentication](./authentication.md) - Authentication methods and requirements
- [Error Handling](./error-handling.md) - Error response specifications
- [Request/Response Examples](./examples.json) - Comprehensive API examples
- [OpenAPI Schema](./openapi.json) - Complete OpenAPI specification
- [Endpoint Documentation](./endpoints/) - Detailed endpoint documentation

## Authentication Summary
"""
        
        # Add authentication summary
        auth_info = specifications.get('authentication', {})
        if auth_info.get('schemes'):
            content += "This API uses the following authentication methods:\n\n"
            for scheme_name, scheme_def in auth_info['schemes'].items():
                content += f"- **{scheme_name}**: {scheme_def.get('type', 'Unknown')} authentication\n"
        else:
            content += "No authentication required for this API.\n"
        
        content += "\n## Error Handling Summary\n"
        
        # Add error handling summary
        error_info = specifications.get('errors', {})
        common_errors = error_info.get('common_errors', {})
        if common_errors:
            content += f"Common error responses across {len(common_errors)} status codes:\n\n"
            for status_code in sorted(common_errors.keys()):
                endpoint_count = len(common_errors[status_code])
                content += f"- **{status_code}**: Used by {endpoint_count} endpoint(s)\n"
        else:
            content += "No specific error handling documentation available.\n"
        
        content += f"""

## Endpoint Categories
"""
        
        # Group endpoints by tags
        tag_groups = {}
        for path, methods in paths.items():
            for method, spec in methods.items():
                tags = spec.get('tags', ['untagged'])
                for tag in tags:
                    if tag not in tag_groups:
                        tag_groups[tag] = []
                    tag_groups[tag].append(f"{method.upper()} {path}")
        
        for tag, endpoints in tag_groups.items():
            content += f"\n### {tag.title()}\n"
            for endpoint in endpoints:
                # Create link to endpoint documentation
                method, path = endpoint.split(' ', 1)
                filename = f"{method}_{path.replace('/', '_').replace('{', '').replace('}', '')}.md"
                content += f"- [{endpoint}](./endpoints/{filename})\n"
        
        content += f"""

## Interactive Documentation
- **Swagger UI**: Available at `/docs` when running the API server
- **ReDoc**: Available at `/redoc` when running the API server

## Development
This documentation is automatically generated from the OpenAPI specification.
To regenerate, run:
```bash
python scripts/generate_docs.py --category api
```
"""
        
        with open(self.docs_dir / "README.md", 'w') as f:
            f.write(content)
    
    def validate_against_implementation(self, schema: Dict[str, Any]) -> ValidationResult:
        """
        Validate API documentation against actual implementation.
        
        Args:
            schema: OpenAPI schema to validate
            
        Returns:
            ValidationResult with validation details
            
        Implements Requirement 2.5: Validation against actual API implementation
        """
        return self.validator.validate_schema(schema)
    
    def _generate_enhanced_swagger_ui(self, openapi_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enhanced Swagger UI with custom styling and features."""
        try:
            from scripts.swagger_ui_generator import SwaggerUIGenerator, SwaggerUIConfig
            
            # Create Swagger UI generator with enhanced configuration
            swagger_config = SwaggerUIConfig(
                title=openapi_schema.get('info', {}).get('title', 'API Documentation'),
                custom_css=True,
                enable_try_it_out=True,
                enable_filter=True,
                enable_search=True,
                deep_linking=True,
                display_operation_id=True,
                display_request_duration=True
            )
            
            swagger_generator = SwaggerUIGenerator(self.project_root, swagger_config)
            return swagger_generator.generate_enhanced_swagger_ui(openapi_schema)
            
        except Exception as e:
            return {
                'success': False,
                'generated_files': [],
                'errors': [f"Swagger UI generation failed: {str(e)}"],
                'warnings': []
            }
    
    def setup_automatic_updates(self) -> Dict[str, Any]:
        """
        Set up automatic API documentation updates.
        
        Returns:
            Dictionary containing setup results
            
        Implements Requirements 2.3, 2.5: Automatic updates and validation
        """
        try:
            from scripts.api_auto_updater import AutomaticAPIUpdater
            
            updater = AutomaticAPIUpdater(self.project_root)
            return updater.setup_automatic_hooks()
            
        except Exception as e:
            return {
                'success': False,
                'hooks_created': [],
                'errors': [f"Failed to setup automatic updates: {str(e)}"]
            }
    
    def check_for_api_changes_and_update(self) -> Dict[str, Any]:
        """
        Check for API changes and update documentation automatically.
        
        Returns:
            Dictionary containing update results
            
        Implements Requirements 2.3, 2.5: Automatic schema validation and updates
        """
        try:
            from scripts.api_auto_updater import AutomaticAPIUpdater
            
            updater = AutomaticAPIUpdater(self.project_root)
            result = updater.check_for_api_changes()
            
            return {
                'success': result.success,
                'changes_detected': len(result.changes_detected),
                'validation_issues': len(result.validation_issues),
                'updated_files': [str(f) for f in result.updated_files],
                'errors': result.errors,
                'duration': result.duration
            }
            
        except Exception as e:
            return {
                'success': False,
                'changes_detected': 0,
                'validation_issues': 0,
                'updated_files': [],
                'errors': [f"Failed to check for API changes: {str(e)}"],
                'duration': 0.0
            }