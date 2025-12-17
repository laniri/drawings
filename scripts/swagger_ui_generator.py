#!/usr/bin/env python3
"""
Enhanced Swagger UI Generator

This module provides enhanced Swagger UI generation capabilities including
custom styling, endpoint testing capabilities with authentication, and
search and filtering for large API specifications.

Implements Requirements 2.2, 2.3 from the comprehensive documentation specification.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SwaggerUIConfig:
    """Configuration for Swagger UI generation."""
    title: str = "API Documentation"
    custom_css: bool = True
    enable_try_it_out: bool = True
    enable_filter: bool = True
    enable_search: bool = True
    show_extensions: bool = True
    show_common_extensions: bool = True
    deep_linking: bool = True
    display_operation_id: bool = True
    default_models_expand_depth: int = 1
    default_model_expand_depth: int = 1
    display_request_duration: bool = True
    doc_expansion: str = "list"  # "list", "full", "none"
    max_displayed_tags: int = 50
    show_mutated_request: bool = True
    supported_submit_methods: List[str] = None
    validator_url: Optional[str] = None
    
    def __post_init__(self):
        if self.supported_submit_methods is None:
            self.supported_submit_methods = ["get", "put", "post", "delete", "options", "head", "patch", "trace"]


class SwaggerUIGenerator:
    """
    Enhanced Swagger UI Generator with custom styling and advanced features.
    
    Provides interactive Swagger UI with custom styling, endpoint testing
    capabilities with authentication, and search and filtering for large
    API specifications.
    
    Implements Requirements 2.2, 2.3 from the comprehensive documentation specification.
    """
    
    def __init__(self, project_root: Path, config: SwaggerUIConfig = None):
        self.project_root = project_root
        self.docs_dir = project_root / "docs" / "api"
        self.swagger_dir = self.docs_dir / "swagger-ui"
        self.config = config or SwaggerUIConfig()
        
        # Ensure directories exist
        self.swagger_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_enhanced_swagger_ui(self, openapi_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate enhanced Swagger UI with custom styling and advanced features.
        
        Args:
            openapi_schema: OpenAPI schema to generate UI for
            
        Returns:
            Dictionary containing generation results and file paths
            
        Implements Requirements 2.2, 2.3: Interactive Swagger UI with custom styling
        """
        result = {
            'success': True,
            'generated_files': [],
            'errors': [],
            'warnings': []
        }
        
        try:
            print("🎨 Generating enhanced Swagger UI...")
            
            # Generate main HTML file
            html_file = self._generate_swagger_html(openapi_schema)
            result['generated_files'].append(html_file)
            
            # Generate custom CSS
            if self.config.custom_css:
                css_file = self._generate_custom_css()
                result['generated_files'].append(css_file)
            
            # Generate JavaScript enhancements
            js_file = self._generate_swagger_enhancements()
            result['generated_files'].append(js_file)
            
            # Generate configuration file
            config_file = self._generate_swagger_config(openapi_schema)
            result['generated_files'].append(config_file)
            
            # Generate authentication helper
            auth_file = self._generate_auth_helper(openapi_schema)
            result['generated_files'].append(auth_file)
            
            print(f"  ✅ Enhanced Swagger UI generated ({len(result['generated_files'])} files)")
            
        except Exception as e:
            result['success'] = False
            result['errors'].append(f"Swagger UI generation failed: {str(e)}")
            print(f"  ❌ Enhanced Swagger UI generation failed: {e}")
        
        return result
    
    def _generate_swagger_html(self, openapi_schema: Dict[str, Any]) -> Path:
        """Generate the main Swagger UI HTML file with enhancements."""
        api_title = openapi_schema.get('info', {}).get('title', 'API Documentation')
        
        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{api_title} - Interactive Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui.css" />
    <link rel="stylesheet" type="text/css" href="./swagger-custom.css" />
    <link rel="icon" type="image/png" href="https://unpkg.com/swagger-ui-dist@5.9.0/favicon-32x32.png" sizes="32x32" />
    <link rel="icon" type="image/png" href="https://unpkg.com/swagger-ui-dist@5.9.0/favicon-16x16.png" sizes="16x16" />
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
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        }}
        .swagger-ui .topbar {{
            background-color: #2c3e50;
            padding: 10px 0;
        }}
        .swagger-ui .topbar .download-url-wrapper {{
            display: none;
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
            position: sticky;
            top: 0;
            background: white;
            padding: 15px 20px;
            border-bottom: 1px solid #e8e8e8;
            z-index: 1000;
        }}
        .search-input {{
            width: 100%;
            padding: 10px 15px;
            border: 2px solid #e8e8e8;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }}
        .search-input:focus {{
            border-color: #667eea;
        }}
        .filter-tags {{
            margin-top: 10px;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}
        .filter-tag {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            padding: 4px 12px;
            border-radius: 15px;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 14px;
        }}
        .filter-tag:hover {{
            background: #e9ecef;
        }}
        .filter-tag.active {{
            background: #667eea;
            color: white;
            border-color: #667eea;
        }}
        .auth-section {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 15px;
            margin: 20px;
            border-radius: 8px;
        }}
        .auth-section h3 {{
            margin-top: 0;
            color: #856404;
        }}
        .auth-input {{
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 5px 0;
        }}
        .auth-button {{
            background: #28a745;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            margin: 5px 5px 5px 0;
        }}
        .auth-button:hover {{
            background: #218838;
        }}
        .clear-auth {{
            background: #dc3545;
        }}
        .clear-auth:hover {{
            background: #c82333;
        }}
    </style>
</head>
<body>
    <div class="custom-header">
        <h1>{api_title}</h1>
        <p>Interactive API Documentation with Enhanced Features</p>
    </div>
    
    <div class="api-info">
        <h2>API Information</h2>
        <p><strong>Version:</strong> {openapi_schema.get('info', {}).get('version', '1.0.0')}</p>
        <p><strong>Description:</strong> {openapi_schema.get('info', {}).get('description', 'No description available')}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="search-container">
        <input type="text" id="api-search" class="search-input" placeholder="🔍 Search endpoints, operations, or descriptions...">
        <div class="filter-tags" id="filter-tags">
            <!-- Tags will be populated by JavaScript -->
        </div>
    </div>
    
    <div class="auth-section" id="auth-section" style="display: none;">
        <h3>🔐 Authentication</h3>
        <div id="auth-controls">
            <!-- Authentication controls will be populated by JavaScript -->
        </div>
    </div>
    
    <div id="swagger-ui"></div>
    
    <script src="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-standalone-preset.js"></script>
    <script src="./swagger-config.js"></script>
    <script src="./swagger-enhancements.js"></script>
    <script>
        // Initialize Swagger UI with enhanced configuration
        window.onload = function() {{
            initializeSwaggerUI();
            initializeEnhancements();
        }};
    </script>
</body>
</html>'''
        
        html_file = self.swagger_dir / "index.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_file
    
    def _generate_custom_css(self) -> Path:
        """Generate custom CSS for enhanced Swagger UI styling."""
        css_content = '''/* Enhanced Swagger UI Custom Styles */

/* Improved color scheme */
.swagger-ui .scheme-container {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 15px;
    margin: 20px 0;
}

/* Enhanced operation styling */
.swagger-ui .opblock {
    border-radius: 8px;
    margin-bottom: 15px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    transition: box-shadow 0.3s ease;
}

.swagger-ui .opblock:hover {
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

/* Method-specific colors */
.swagger-ui .opblock.opblock-get {
    border-color: #61affe;
    background: rgba(97, 175, 254, 0.1);
}

.swagger-ui .opblock.opblock-post {
    border-color: #49cc90;
    background: rgba(73, 204, 144, 0.1);
}

.swagger-ui .opblock.opblock-put {
    border-color: #fca130;
    background: rgba(252, 161, 48, 0.1);
}

.swagger-ui .opblock.opblock-delete {
    border-color: #f93e3e;
    background: rgba(249, 62, 62, 0.1);
}

/* Enhanced parameter styling */
.swagger-ui .parameters-col_description p {
    margin: 0;
    color: #666;
}

.swagger-ui .parameter__name {
    font-weight: 600;
    color: #333;
}

.swagger-ui .parameter__type {
    font-size: 12px;
    color: #999;
    font-family: monospace;
}

/* Improved response styling */
.swagger-ui .responses-inner h4 {
    font-size: 14px;
    margin: 10px 0 5px 0;
}

.swagger-ui .response-col_status {
    font-family: monospace;
    font-weight: bold;
}

/* Enhanced code blocks */
.swagger-ui .highlight-code {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 4px;
    padding: 15px;
}

.swagger-ui .microlight {
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    font-size: 13px;
    line-height: 1.4;
}

/* Try it out button styling */
.swagger-ui .btn.try-out__btn {
    background: #667eea;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: 500;
    transition: background-color 0.3s;
}

.swagger-ui .btn.try-out__btn:hover {
    background: #5a6fd8;
}

/* Execute button styling */
.swagger-ui .btn.execute {
    background: #28a745;
    border-color: #28a745;
    color: white;
    font-weight: 500;
    padding: 10px 20px;
    border-radius: 4px;
}

.swagger-ui .btn.execute:hover {
    background: #218838;
    border-color: #1e7e34;
}

/* Model styling */
.swagger-ui .model-box {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 4px;
    padding: 15px;
    margin: 10px 0;
}

.swagger-ui .model .property {
    padding: 8px 0;
    border-bottom: 1px solid #f0f0f0;
}

.swagger-ui .model .property:last-child {
    border-bottom: none;
}

/* Authentication section styling */
.swagger-ui .auth-wrapper {
    background: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 8px;
    padding: 20px;
    margin: 20px 0;
}

.swagger-ui .auth-wrapper h4 {
    color: #856404;
    margin-top: 0;
}

/* Responsive design improvements */
@media (max-width: 768px) {
    .swagger-ui .wrapper {
        padding: 0 10px;
    }
    
    .custom-header {
        padding: 15px 10px;
    }
    
    .custom-header h1 {
        font-size: 2em;
    }
    
    .api-info {
        margin: 10px;
        padding: 15px;
    }
    
    .search-container {
        padding: 10px;
    }
}

/* Loading animation */
.swagger-ui .loading-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 200px;
}

.swagger-ui .loading-spinner {
    border: 4px solid #f3f3f3;
    border-top: 4px solid #667eea;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Enhanced error styling */
.swagger-ui .errors-wrapper {
    background: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
    border-radius: 4px;
    padding: 15px;
    margin: 10px 0;
}

/* Improved table styling */
.swagger-ui table {
    border-collapse: collapse;
    width: 100%;
}

.swagger-ui table th,
.swagger-ui table td {
    border: 1px solid #e9ecef;
    padding: 12px 8px;
    text-align: left;
}

.swagger-ui table th {
    background: #f8f9fa;
    font-weight: 600;
}

/* Enhanced download button */
.swagger-ui .download-contents {
    background: #6c757d;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-size: 14px;
    cursor: pointer;
    transition: background-color 0.3s;
}

.swagger-ui .download-contents:hover {
    background: #5a6268;
}'''
        
        css_file = self.swagger_dir / "swagger-custom.css"
        with open(css_file, 'w', encoding='utf-8') as f:
            f.write(css_content)
        
        return css_file
    
    def _generate_swagger_enhancements(self) -> Path:
        """Generate JavaScript enhancements for search, filtering, and authentication."""
        js_content = '''// Enhanced Swagger UI JavaScript Enhancements

// Global variables
let swaggerUI = null;
let originalSpec = null;
let currentFilter = '';
let activeTagFilters = new Set();

// Initialize enhanced features
function initializeEnhancements() {
    setupSearch();
    setupTagFilters();
    setupAuthentication();
    setupKeyboardShortcuts();
    setupAdvancedFeatures();
}

// Search functionality
function setupSearch() {
    const searchInput = document.getElementById('api-search');
    if (!searchInput) return;
    
    searchInput.addEventListener('input', debounce(handleSearch, 300));
    searchInput.addEventListener('keydown', handleSearchKeydown);
}

function handleSearch(event) {
    const query = event.target.value.toLowerCase().trim();
    currentFilter = query;
    
    if (!query && activeTagFilters.size === 0) {
        // Reset to original spec
        if (originalSpec && swaggerUI) {
            swaggerUI.specActions.updateSpec(JSON.stringify(originalSpec));
        }
        return;
    }
    
    filterEndpoints();
}

function handleSearchKeydown(event) {
    if (event.key === 'Escape') {
        event.target.value = '';
        currentFilter = '';
        filterEndpoints();
    }
}

function filterEndpoints() {
    if (!originalSpec || !swaggerUI) return;
    
    const filteredSpec = JSON.parse(JSON.stringify(originalSpec));
    const filteredPaths = {};
    
    for (const [path, methods] of Object.entries(originalSpec.paths || {})) {
        const filteredMethods = {};
        
        for (const [method, operation] of Object.entries(methods)) {
            if (shouldIncludeOperation(path, method, operation)) {
                filteredMethods[method] = operation;
            }
        }
        
        if (Object.keys(filteredMethods).length > 0) {
            filteredPaths[path] = filteredMethods;
        }
    }
    
    filteredSpec.paths = filteredPaths;
    swaggerUI.specActions.updateSpec(JSON.stringify(filteredSpec));
    
    // Update result count
    updateSearchResults(Object.keys(filteredPaths).length);
}

function shouldIncludeOperation(path, method, operation) {
    // Check tag filters
    if (activeTagFilters.size > 0) {
        const operationTags = operation.tags || [];
        const hasMatchingTag = operationTags.some(tag => activeTagFilters.has(tag));
        if (!hasMatchingTag) return false;
    }
    
    // Check search query
    if (!currentFilter) return true;
    
    const searchableText = [
        path,
        method,
        operation.summary || '',
        operation.description || '',
        operation.operationId || '',
        ...(operation.tags || [])
    ].join(' ').toLowerCase();
    
    return searchableText.includes(currentFilter);
}

function updateSearchResults(count) {
    const searchInput = document.getElementById('api-search');
    if (!searchInput) return;
    
    if (currentFilter || activeTagFilters.size > 0) {
        searchInput.placeholder = `🔍 Found ${count} matching endpoints`;
    } else {
        searchInput.placeholder = '🔍 Search endpoints, operations, or descriptions...';
    }
}

// Tag filtering
function setupTagFilters() {
    if (!originalSpec) return;
    
    const tags = extractTags(originalSpec);
    renderTagFilters(tags);
}

function extractTags(spec) {
    const tagSet = new Set();
    
    for (const methods of Object.values(spec.paths || {})) {
        for (const operation of Object.values(methods)) {
            if (operation.tags) {
                operation.tags.forEach(tag => tagSet.add(tag));
            }
        }
    }
    
    return Array.from(tagSet).sort();
}

function renderTagFilters(tags) {
    const container = document.getElementById('filter-tags');
    if (!container) return;
    
    container.innerHTML = '';
    
    // Add "All" filter
    const allTag = createTagFilter('All', () => {
        activeTagFilters.clear();
        updateTagFilterUI();
        filterEndpoints();
    });
    allTag.classList.add('active');
    container.appendChild(allTag);
    
    // Add individual tag filters
    tags.forEach(tag => {
        const tagElement = createTagFilter(tag, () => toggleTagFilter(tag));
        container.appendChild(tagElement);
    });
}

function createTagFilter(text, onClick) {
    const element = document.createElement('span');
    element.className = 'filter-tag';
    element.textContent = text;
    element.addEventListener('click', onClick);
    return element;
}

function toggleTagFilter(tag) {
    if (activeTagFilters.has(tag)) {
        activeTagFilters.delete(tag);
    } else {
        activeTagFilters.add(tag);
    }
    
    updateTagFilterUI();
    filterEndpoints();
}

function updateTagFilterUI() {
    const tags = document.querySelectorAll('.filter-tag');
    tags.forEach(tagElement => {
        const tagText = tagElement.textContent;
        
        if (tagText === 'All') {
            tagElement.classList.toggle('active', activeTagFilters.size === 0);
        } else {
            tagElement.classList.toggle('active', activeTagFilters.has(tagText));
        }
    });
}

// Authentication helpers
function setupAuthentication() {
    if (!originalSpec) return;
    
    const securitySchemes = originalSpec.components?.securitySchemes;
    if (!securitySchemes) return;
    
    renderAuthenticationControls(securitySchemes);
    document.getElementById('auth-section').style.display = 'block';
}

function renderAuthenticationControls(securitySchemes) {
    const container = document.getElementById('auth-controls');
    if (!container) return;
    
    container.innerHTML = '';
    
    for (const [schemeName, scheme] of Object.entries(securitySchemes)) {
        const authControl = createAuthControl(schemeName, scheme);
        container.appendChild(authControl);
    }
    
    // Add clear all button
    const clearButton = document.createElement('button');
    clearButton.className = 'auth-button clear-auth';
    clearButton.textContent = 'Clear All Authentication';
    clearButton.addEventListener('click', clearAllAuth);
    container.appendChild(clearButton);
}

function createAuthControl(schemeName, scheme) {
    const container = document.createElement('div');
    container.style.marginBottom = '15px';
    
    const label = document.createElement('label');
    label.textContent = `${schemeName} (${scheme.type})`;
    label.style.display = 'block';
    label.style.fontWeight = 'bold';
    label.style.marginBottom = '5px';
    
    const input = document.createElement('input');
    input.className = 'auth-input';
    input.type = scheme.type === 'http' && scheme.scheme === 'basic' ? 'password' : 'text';
    input.placeholder = getAuthPlaceholder(scheme);
    input.id = `auth-${schemeName}`;
    
    const button = document.createElement('button');
    button.className = 'auth-button';
    button.textContent = 'Apply';
    button.addEventListener('click', () => applyAuth(schemeName, scheme, input.value));
    
    container.appendChild(label);
    container.appendChild(input);
    container.appendChild(button);
    
    return container;
}

function getAuthPlaceholder(scheme) {
    switch (scheme.type) {
        case 'http':
            if (scheme.scheme === 'bearer') {
                return 'Enter your bearer token';
            } else if (scheme.scheme === 'basic') {
                return 'Enter username:password';
            }
            return 'Enter authentication value';
        case 'apiKey':
            return `Enter your API key (${scheme.in}: ${scheme.name})`;
        case 'oauth2':
            return 'OAuth2 flow will be handled by Swagger UI';
        default:
            return 'Enter authentication value';
    }
}

function applyAuth(schemeName, scheme, value) {
    if (!swaggerUI || !value.trim()) return;
    
    try {
        if (scheme.type === 'http' && scheme.scheme === 'bearer') {
            swaggerUI.preauthorizeApiKey(schemeName, `Bearer ${value.trim()}`);
        } else if (scheme.type === 'http' && scheme.scheme === 'basic') {
            const encoded = btoa(value.trim());
            swaggerUI.preauthorizeBasic(schemeName, value.trim().split(':')[0], value.trim().split(':')[1]);
        } else if (scheme.type === 'apiKey') {
            swaggerUI.preauthorizeApiKey(schemeName, value.trim());
        }
        
        showNotification(`Authentication applied for ${schemeName}`, 'success');
    } catch (error) {
        showNotification(`Failed to apply authentication: ${error.message}`, 'error');
    }
}

function clearAllAuth() {
    if (!swaggerUI) return;
    
    // Clear all authentication
    const authInputs = document.querySelectorAll('.auth-input');
    authInputs.forEach(input => input.value = '');
    
    // This is a workaround since Swagger UI doesn't have a direct clear method
    location.reload();
}

// Keyboard shortcuts
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (event) => {
        // Ctrl/Cmd + K to focus search
        if ((event.ctrlKey || event.metaKey) && event.key === 'k') {
            event.preventDefault();
            const searchInput = document.getElementById('api-search');
            if (searchInput) {
                searchInput.focus();
                searchInput.select();
            }
        }
        
        // Escape to clear search
        if (event.key === 'Escape') {
            const searchInput = document.getElementById('api-search');
            if (searchInput && document.activeElement === searchInput) {
                searchInput.value = '';
                currentFilter = '';
                filterEndpoints();
                searchInput.blur();
            }
        }
    });
}

// Advanced features
function setupAdvancedFeatures() {
    // Add copy buttons to code blocks
    addCopyButtons();
    
    // Add expand/collapse all buttons
    addExpandCollapseButtons();
    
    // Add export functionality
    addExportFunctionality();
}

function addCopyButtons() {
    // This will be called after Swagger UI renders
    setTimeout(() => {
        const codeBlocks = document.querySelectorAll('.highlight-code, .microlight');
        codeBlocks.forEach(block => {
            if (block.querySelector('.copy-button')) return; // Already has copy button
            
            const copyButton = document.createElement('button');
            copyButton.className = 'copy-button';
            copyButton.textContent = 'Copy';
            copyButton.style.cssText = `
                position: absolute;
                top: 5px;
                right: 5px;
                background: #667eea;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 3px;
                font-size: 12px;
                cursor: pointer;
            `;
            
            copyButton.addEventListener('click', () => {
                navigator.clipboard.writeText(block.textContent).then(() => {
                    copyButton.textContent = 'Copied!';
                    setTimeout(() => copyButton.textContent = 'Copy', 2000);
                });
            });
            
            block.style.position = 'relative';
            block.appendChild(copyButton);
        });
    }, 2000);
}

function addExpandCollapseButtons() {
    const container = document.querySelector('.swagger-ui');
    if (!container) return;
    
    const buttonContainer = document.createElement('div');
    buttonContainer.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
        display: flex;
        gap: 10px;
    `;
    
    const expandAllBtn = createActionButton('Expand All', () => {
        document.querySelectorAll('.opblock-summary').forEach(summary => {
            if (!summary.parentElement.classList.contains('is-open')) {
                summary.click();
            }
        });
    });
    
    const collapseAllBtn = createActionButton('Collapse All', () => {
        document.querySelectorAll('.opblock-summary').forEach(summary => {
            if (summary.parentElement.classList.contains('is-open')) {
                summary.click();
            }
        });
    });
    
    buttonContainer.appendChild(expandAllBtn);
    buttonContainer.appendChild(collapseAllBtn);
    document.body.appendChild(buttonContainer);
}

function createActionButton(text, onClick) {
    const button = document.createElement('button');
    button.textContent = text;
    button.style.cssText = `
        background: #667eea;
        color: white;
        border: none;
        padding: 8px 12px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 14px;
    `;
    button.addEventListener('click', onClick);
    return button;
}

function addExportFunctionality() {
    // Add export button to download filtered spec
    const exportBtn = createActionButton('Export Filtered', () => {
        const currentSpec = swaggerUI.getState().spec.json;
        const blob = new Blob([JSON.stringify(currentSpec, null, 2)], {
            type: 'application/json'
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'filtered-api-spec.json';
        a.click();
        URL.revokeObjectURL(url);
    });
    
    // Add to existing button container or create new one
    const existingContainer = document.querySelector('div[style*="position: fixed"]');
    if (existingContainer) {
        existingContainer.appendChild(exportBtn);
    }
}

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: ${type === 'error' ? '#dc3545' : '#28a745'};
        color: white;
        padding: 12px 20px;
        border-radius: 4px;
        z-index: 10000;
        font-size: 14px;
    `;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// Store original spec when Swagger UI loads
function storeOriginalSpec(spec) {
    originalSpec = spec;
    setupTagFilters();
    setupAuthentication();
}'''
        
        js_file = self.swagger_dir / "swagger-enhancements.js"
        with open(js_file, 'w', encoding='utf-8') as f:
            f.write(js_content)
        
        return js_file
    
    def _generate_swagger_config(self, openapi_schema: Dict[str, Any]) -> Path:
        """Generate Swagger UI configuration JavaScript."""
        config_content = f'''// Swagger UI Configuration

function initializeSwaggerUI() {{
    // Store the original spec
    const spec = {json.dumps(openapi_schema, indent=2)};
    storeOriginalSpec(spec);
    
    // Initialize Swagger UI with enhanced configuration
    window.swaggerUI = SwaggerUIBundle({{
        url: './openapi.json',
        dom_id: '#swagger-ui',
        deepLinking: {str(self.config.deep_linking).lower()},
        presets: [
            SwaggerUIBundle.presets.apis,
            SwaggerUIStandalonePreset
        ],
        plugins: [
            SwaggerUIBundle.plugins.DownloadUrl
        ],
        layout: "StandaloneLayout",
        
        // Enhanced configuration options
        displayOperationId: {str(self.config.display_operation_id).lower()},
        displayRequestDuration: {str(self.config.display_request_duration).lower()},
        docExpansion: "{self.config.doc_expansion}",
        filter: {str(self.config.enable_filter).lower()},
        showExtensions: {str(self.config.show_extensions).lower()},
        showCommonExtensions: {str(self.config.show_common_extensions).lower()},
        maxDisplayedTags: {self.config.max_displayed_tags},
        showMutatedRequest: {str(self.config.show_mutated_request).lower()},
        supportedSubmitMethods: {json.dumps(self.config.supported_submit_methods)},
        
        // Default model expansion
        defaultModelsExpandDepth: {self.config.default_models_expand_depth},
        defaultModelExpandDepth: {self.config.default_model_expand_depth},
        
        // Validator URL (set to null to disable validation)
        validatorUrl: {json.dumps(self.config.validator_url)},
        
        // Try it out enabled by default
        tryItOutEnabled: {str(self.config.enable_try_it_out).lower()},
        
        // Request interceptor for authentication
        requestInterceptor: function(request) {{
            // Add custom headers or modify requests here
            console.log('Request interceptor:', request);
            return request;
        }},
        
        // Response interceptor for handling responses
        responseInterceptor: function(response) {{
            // Handle responses here
            console.log('Response interceptor:', response);
            return response;
        }},
        
        // Error handling
        onComplete: function() {{
            console.log('Swagger UI loaded successfully');
            
            // Initialize enhancements after UI is ready
            setTimeout(() => {{
                setupAdvancedFeatures();
            }}, 1000);
        }},
        
        onFailure: function(error) {{
            console.error('Swagger UI failed to load:', error);
            showNotification('Failed to load API documentation', 'error');
        }}
    }});
    
    // Store reference globally
    window.swaggerUI = window.swaggerUI;
}}

// Configuration constants
const SWAGGER_CONFIG = {{
    SEARCH_DEBOUNCE_MS: 300,
    NOTIFICATION_DURATION_MS: 3000,
    COPY_BUTTON_RESET_MS: 2000,
    ENHANCEMENT_INIT_DELAY_MS: 1000,
    
    // Feature flags
    ENABLE_KEYBOARD_SHORTCUTS: true,
    ENABLE_COPY_BUTTONS: true,
    ENABLE_EXPORT_FUNCTIONALITY: true,
    ENABLE_EXPAND_COLLAPSE: true,
    
    // Styling
    PRIMARY_COLOR: '#667eea',
    SUCCESS_COLOR: '#28a745',
    ERROR_COLOR: '#dc3545',
    WARNING_COLOR: '#ffc107'
}};'''
        
        config_file = self.swagger_dir / "swagger-config.js"
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        return config_file
    
    def _generate_auth_helper(self, openapi_schema: Dict[str, Any]) -> Path:
        """Generate authentication helper documentation."""
        security_schemes = openapi_schema.get('components', {}).get('securitySchemes', {})
        
        auth_content = f'''# Authentication Guide

This document provides guidance on how to authenticate with the API using the interactive Swagger UI.

## Available Authentication Methods

'''
        
        if not security_schemes:
            auth_content += "No authentication methods are configured for this API.\n"
        else:
            for scheme_name, scheme in security_schemes.items():
                auth_content += f"### {scheme_name}\n\n"
                auth_content += f"**Type**: {scheme.get('type', 'Unknown')}\n\n"
                
                if 'description' in scheme:
                    auth_content += f"**Description**: {scheme['description']}\n\n"
                
                scheme_type = scheme.get('type', '')
                
                if scheme_type == 'http':
                    http_scheme = scheme.get('scheme', '')
                    auth_content += f"**HTTP Scheme**: {http_scheme}\n\n"
                    
                    if http_scheme == 'bearer':
                        bearer_format = scheme.get('bearerFormat', 'JWT')
                        auth_content += f"**Bearer Format**: {bearer_format}\n\n"
                        auth_content += "**How to use in Swagger UI**:\n"
                        auth_content += "1. Click the 'Authorize' button at the top of the page\n"
                        auth_content += "2. Enter your bearer token in the format: `your_token_here`\n"
                        auth_content += "3. Click 'Authorize' to apply the token to all requests\n\n"
                        auth_content += "**Alternative**: Use the authentication section above the API documentation to quickly apply your token.\n\n"
                    
                    elif http_scheme == 'basic':
                        auth_content += "**How to use in Swagger UI**:\n"
                        auth_content += "1. Click the 'Authorize' button at the top of the page\n"
                        auth_content += "2. Enter your username and password\n"
                        auth_content += "3. Click 'Authorize' to apply the credentials to all requests\n\n"
                
                elif scheme_type == 'apiKey':
                    key_location = scheme.get('in', 'header')
                    key_name = scheme.get('name', 'X-API-Key')
                    auth_content += f"**Location**: {key_location}\n"
                    auth_content += f"**Parameter Name**: {key_name}\n\n"
                    auth_content += "**How to use in Swagger UI**:\n"
                    auth_content += "1. Click the 'Authorize' button at the top of the page\n"
                    auth_content += f"2. Enter your API key in the {key_name} field\n"
                    auth_content += "3. Click 'Authorize' to apply the key to all requests\n\n"
                
                elif scheme_type == 'oauth2':
                    flows = scheme.get('flows', {})
                    auth_content += "**OAuth 2.0 Flows**:\n"
                    for flow_name, flow_config in flows.items():
                        auth_content += f"- **{flow_name}**: {flow_config.get('authorizationUrl', 'N/A')}\n"
                    auth_content += "\n**How to use in Swagger UI**:\n"
                    auth_content += "1. Click the 'Authorize' button at the top of the page\n"
                    auth_content += "2. Follow the OAuth 2.0 flow to obtain authorization\n"
                    auth_content += "3. The token will be automatically applied to requests\n\n"
        
        auth_content += '''## Quick Authentication Tips

### Using the Enhanced Authentication Section
This Swagger UI includes an enhanced authentication section above the API documentation:

1. **Bearer Token**: Enter your token directly (without "Bearer " prefix)
2. **API Key**: Enter your API key value
3. **Basic Auth**: Enter in format `username:password`
4. Click "Apply" to authenticate
5. Use "Clear All Authentication" to remove all credentials

### Keyboard Shortcuts
- **Ctrl/Cmd + K**: Focus the search box
- **Escape**: Clear search or close focused elements

### Testing Endpoints
1. Find the endpoint you want to test
2. Click "Try it out" button
3. Fill in required parameters
4. Click "Execute" to send the request
5. View the response below

### Troubleshooting Authentication
- **401 Unauthorized**: Check that your credentials are correct and properly formatted
- **403 Forbidden**: You may not have permission to access this resource
- **Token Expired**: Refresh your token and re-authenticate

### Security Best Practices
- Never share your API keys or tokens
- Use HTTPS in production environments
- Regularly rotate your credentials
- Monitor API usage for suspicious activity

## Advanced Features

### Search and Filtering
- Use the search box to find specific endpoints
- Click on tag filters to show only endpoints with specific tags
- Combine search and tag filters for precise results

### Export Functionality
- Click "Export Filtered" to download the current filtered API specification
- Use "Expand All" / "Collapse All" to control endpoint visibility

### Copy Code Examples
- Hover over code blocks to see copy buttons
- Click to copy request/response examples to clipboard
'''
        
        auth_file = self.swagger_dir / "authentication-guide.md"
        with open(auth_file, 'w', encoding='utf-8') as f:
            f.write(auth_content)
        
        return auth_file
    
    def generate_swagger_ui_with_authentication(self, openapi_schema: Dict[str, Any], 
                                             auth_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate Swagger UI with enhanced authentication capabilities.
        
        Args:
            openapi_schema: OpenAPI schema to generate UI for
            auth_config: Additional authentication configuration
            
        Returns:
            Dictionary containing generation results
            
        Implements Requirement 2.2: Endpoint testing capabilities with authentication
        """
        # Update configuration with auth settings if provided
        if auth_config:
            for key, value in auth_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        return self.generate_enhanced_swagger_ui(openapi_schema)
    
    def generate_searchable_swagger_ui(self, openapi_schema: Dict[str, Any], 
                                     search_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate Swagger UI with enhanced search and filtering capabilities.
        
        Args:
            openapi_schema: OpenAPI schema to generate UI for
            search_config: Additional search configuration
            
        Returns:
            Dictionary containing generation results
            
        Implements Requirement 2.3: Search and filtering for large API specifications
        """
        # Update configuration with search settings if provided
        if search_config:
            self.config.enable_search = search_config.get('enable_search', True)
            self.config.enable_filter = search_config.get('enable_filter', True)
            self.config.max_displayed_tags = search_config.get('max_displayed_tags', 50)
        
        return self.generate_enhanced_swagger_ui(openapi_schema)