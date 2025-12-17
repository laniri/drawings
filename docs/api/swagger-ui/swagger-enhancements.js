// Enhanced Swagger UI JavaScript Enhancements

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
}