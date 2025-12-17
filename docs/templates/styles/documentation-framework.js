/**
 * Documentation Framework JavaScript
 * Handles theme switching, responsive navigation, and accessibility features
 */

class DocumentationFramework {
  constructor() {
    this.currentTheme = this.getStoredTheme() || this.getPreferredTheme();
    this.mobileBreakpoint = 768;
    this.tabletBreakpoint = 1024;
    
    this.init();
  }

  init() {
    this.setTheme(this.currentTheme);
    this.setupThemeToggle();
    this.setupMobileNavigation();
    this.setupSearch();
    this.setupTableOfContents();
    this.setupScrollSpy();
    this.setupAccessibility();
    this.setupResponsiveImages();
    this.setupCodeCopyButtons();
    
    // Listen for system theme changes
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
      if (!this.getStoredTheme()) {
        this.setTheme(e.matches ? 'dark' : 'light');
      }
    });

    // Listen for resize events
    window.addEventListener('resize', this.debounce(() => {
      this.handleResize();
    }, 250));
  }

  // ===== THEME MANAGEMENT =====
  
  getStoredTheme() {
    return localStorage.getItem('documentation-theme');
  }

  getPreferredTheme() {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  setTheme(theme) {
    this.currentTheme = theme;
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('documentation-theme', theme);
    
    // Update theme toggle button
    const themeToggle = document.querySelector('.theme-toggle');
    if (themeToggle) {
      themeToggle.innerHTML = theme === 'dark' ? '☀️' : '🌙';
      themeToggle.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`);
    }
  }

  setupThemeToggle() {
    const themeToggle = document.querySelector('.theme-toggle');
    if (themeToggle) {
      themeToggle.addEventListener('click', () => {
        const newTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
        this.setTheme(newTheme);
      });
    }
  }

  // ===== MOBILE NAVIGATION =====
  
  setupMobileNavigation() {
    const mobileToggle = document.querySelector('.mobile-nav-toggle');
    const mobileSidebar = document.querySelector('.mobile-sidebar');
    const mobileOverlay = document.querySelector('.mobile-overlay');
    
    if (mobileToggle && mobileSidebar && mobileOverlay) {
      mobileToggle.addEventListener('click', () => {
        this.toggleMobileNav();
      });
      
      mobileOverlay.addEventListener('click', () => {
        this.closeMobileNav();
      });
      
      // Close on escape key
      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && mobileSidebar.classList.contains('open')) {
          this.closeMobileNav();
        }
      });
    }
  }

  toggleMobileNav() {
    const mobileSidebar = document.querySelector('.mobile-sidebar');
    const mobileOverlay = document.querySelector('.mobile-overlay');
    
    if (mobileSidebar && mobileOverlay) {
      const isOpen = mobileSidebar.classList.contains('open');
      
      if (isOpen) {
        this.closeMobileNav();
      } else {
        this.openMobileNav();
      }
    }
  }

  openMobileNav() {
    const mobileSidebar = document.querySelector('.mobile-sidebar');
    const mobileOverlay = document.querySelector('.mobile-overlay');
    const mobileToggle = document.querySelector('.mobile-nav-toggle');
    
    if (mobileSidebar && mobileOverlay) {
      mobileSidebar.classList.add('open');
      mobileOverlay.classList.add('open');
      document.body.style.overflow = 'hidden';
      
      if (mobileToggle) {
        mobileToggle.setAttribute('aria-expanded', 'true');
      }
      
      // Focus first focusable element in sidebar
      const firstFocusable = mobileSidebar.querySelector('a, button, input, [tabindex]:not([tabindex="-1"])');
      if (firstFocusable) {
        firstFocusable.focus();
      }
    }
  }

  closeMobileNav() {
    const mobileSidebar = document.querySelector('.mobile-sidebar');
    const mobileOverlay = document.querySelector('.mobile-overlay');
    const mobileToggle = document.querySelector('.mobile-nav-toggle');
    
    if (mobileSidebar && mobileOverlay) {
      mobileSidebar.classList.remove('open');
      mobileOverlay.classList.remove('open');
      document.body.style.overflow = '';
      
      if (mobileToggle) {
        mobileToggle.setAttribute('aria-expanded', 'false');
        mobileToggle.focus();
      }
    }
  }

  // ===== SEARCH FUNCTIONALITY =====
  
  setupSearch() {
    const searchInput = document.querySelector('#doc-search');
    const searchResults = document.querySelector('#search-results');
    const searchSuggestions = document.querySelector('#search-suggestions');
    
    if (searchInput && searchResults) {
      let searchTimeout;
      let suggestionTimeout;
      
      // Main search functionality
      searchInput.addEventListener('input', (e) => {
        const query = e.target.value.trim();
        
        clearTimeout(searchTimeout);
        clearTimeout(suggestionTimeout);
        
        if (query.length === 0) {
          this.clearSearch();
          return;
        }
        
        // Show suggestions for short queries
        if (query.length >= 2 && query.length < 4) {
          suggestionTimeout = setTimeout(() => {
            this.showSearchSuggestions(query);
          }, 200);
        }
        
        // Perform full search for longer queries
        if (query.length >= 3) {
          searchTimeout = setTimeout(() => {
            this.performAdvancedSearch(query);
          }, 300);
        }
      });
      
      // Keyboard navigation
      searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
          this.clearSearch();
        } else if (e.key === 'ArrowDown') {
          e.preventDefault();
          this.navigateSearchResults('down');
        } else if (e.key === 'ArrowUp') {
          e.preventDefault();
          this.navigateSearchResults('up');
        } else if (e.key === 'Enter') {
          e.preventDefault();
          this.selectSearchResult();
        }
      });
      
      // Close search results when clicking outside
      document.addEventListener('click', (e) => {
        if (!searchInput.contains(e.target) && 
            !searchResults.contains(e.target) && 
            !searchSuggestions.contains(e.target)) {
          this.clearSearch();
        }
      });
      
      // Focus search with keyboard shortcut
      document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
          e.preventDefault();
          searchInput.focus();
        }
      });
    }
  }

  performSearch(query) {
    const searchResults = document.querySelector('.search-results');
    
    if (!query.trim()) {
      this.clearSearch();
      return;
    }
    
    // Simple search implementation - in production, this would use a search index
    const results = this.searchContent(query);
    this.displaySearchResults(results);
  }

  searchContent(query) {
    const content = document.querySelectorAll('h1, h2, h3, h4, h5, h6, p');
    const results = [];
    const queryLower = query.toLowerCase();
    
    content.forEach((element) => {
      const text = element.textContent.toLowerCase();
      if (text.includes(queryLower)) {
        results.push({
          title: element.textContent,
          type: element.tagName.toLowerCase(),
          element: element,
          snippet: this.createSnippet(element.textContent, query)
        });
      }
    });
    
    return results.slice(0, 10); // Limit to 10 results
  }

  createSnippet(text, query) {
    const index = text.toLowerCase().indexOf(query.toLowerCase());
    const start = Math.max(0, index - 50);
    const end = Math.min(text.length, index + query.length + 50);
    
    let snippet = text.slice(start, end);
    if (start > 0) snippet = '...' + snippet;
    if (end < text.length) snippet = snippet + '...';
    
    // Highlight the query
    const regex = new RegExp(`(${query})`, 'gi');
    snippet = snippet.replace(regex, '<mark>$1</mark>');
    
    return snippet;
  }

  displaySearchResults(results) {
    const searchResults = document.querySelector('.search-results');
    
    if (!searchResults) return;
    
    if (results.length === 0) {
      searchResults.innerHTML = '<div class="search-result-item">No results found</div>';
    } else {
      searchResults.innerHTML = results.map(result => `
        <div class="search-result-item" data-target="${result.element.id || ''}">
          <div class="search-result-title">${result.title}</div>
          <div class="search-result-snippet">${result.snippet}</div>
        </div>
      `).join('');
      
      // Add click handlers
      searchResults.querySelectorAll('.search-result-item').forEach(item => {
        item.addEventListener('click', () => {
          const targetId = item.dataset.target;
          if (targetId) {
            const targetElement = document.getElementById(targetId);
            if (targetElement) {
              targetElement.scrollIntoView({ behavior: 'smooth' });
              this.clearSearch();
            }
          }
        });
      });
    }
    
    searchResults.style.display = 'block';
  }

  async performAdvancedSearch(query) {
    const searchResults = document.querySelector('#search-results');
    const searchContent = searchResults.querySelector('.search-content');
    const searchLoading = searchResults.querySelector('.search-loading');
    const searchFacets = searchResults.querySelector('.search-facets');
    
    if (!searchResults || !searchContent) return;
    
    // Show loading state
    searchLoading.style.display = 'block';
    searchContent.innerHTML = '';
    searchResults.style.display = 'block';
    
    try {
      // In a real implementation, this would call the search API
      // For now, we'll simulate with client-side search
      const results = await this.clientSideSearch(query);
      
      searchLoading.style.display = 'none';
      
      if (results.results.length === 0) {
        searchContent.innerHTML = `
          <div class="search-no-results">
            <p>No results found for "${query}"</p>
            ${results.suggestions.length > 0 ? `
              <p>Did you mean:</p>
              <ul class="search-suggestions-list">
                ${results.suggestions.map(suggestion => 
                  `<li><a href="#" onclick="document.getElementById('doc-search').value='${suggestion}'; this.performAdvancedSearch('${suggestion}'); return false;">${suggestion}</a></li>`
                ).join('')}
              </ul>
            ` : ''}
          </div>
        `;
      } else {
        searchContent.innerHTML = `
          <div class="search-results-header">
            <span class="search-count">${results.total_count} results found in ${results.query_time}ms</span>
          </div>
          <div class="search-results-list">
            ${results.results.map((result, index) => `
              <div class="search-result-item" data-index="${index}">
                <h4 class="search-result-title">
                  <a href="${result.document.url}">${result.document.title}</a>
                </h4>
                <div class="search-result-meta">
                  <span class="search-result-type">${result.document.doc_type}</span>
                  <span class="search-result-score">Score: ${result.score.toFixed(2)}</span>
                </div>
                <div class="search-result-snippet">${result.snippet}</div>
                ${result.highlights.length > 0 ? `
                  <div class="search-result-highlights">
                    ${result.highlights.map(highlight => `<div class="highlight">${highlight}</div>`).join('')}
                  </div>
                ` : ''}
              </div>
            `).join('')}
          </div>
        `;
        
        // Show facets if available
        if (results.facets && Object.keys(results.facets).length > 0) {
          this.displaySearchFacets(results.facets);
          searchFacets.style.display = 'block';
        }
      }
      
    } catch (error) {
      searchLoading.style.display = 'none';
      searchContent.innerHTML = `
        <div class="search-error">
          <p>Search error: ${error.message}</p>
        </div>
      `;
    }
  }
  
  async clientSideSearch(query) {
    // Simulate API call with client-side search
    // In production, this would call the actual search API
    const results = this.searchContent(query);
    
    return {
      results: results,
      total_count: results.length,
      query_time: Math.random() * 100 + 50, // Simulate query time
      facets: this.generateFacets(results),
      suggestions: results.length === 0 ? this.generateSuggestions(query) : []
    };
  }
  
  generateFacets(results) {
    const facets = {
      doc_types: {},
      tags: {}
    };
    
    results.forEach(result => {
      // Count document types
      const docType = result.document.doc_type || 'general';
      facets.doc_types[docType] = (facets.doc_types[docType] || 0) + 1;
      
      // Count tags
      if (result.document.tags) {
        result.document.tags.forEach(tag => {
          facets.tags[tag] = (facets.tags[tag] || 0) + 1;
        });
      }
    });
    
    return facets;
  }
  
  generateSuggestions(query) {
    // Simple suggestion generation
    const commonTerms = ['api', 'architecture', 'algorithm', 'workflow', 'interface', 'deployment'];
    const queryLower = query.toLowerCase();
    
    return commonTerms.filter(term => 
      term.includes(queryLower) || queryLower.includes(term)
    ).slice(0, 3);
  }
  
  displaySearchFacets(facets) {
    const typeFacets = document.querySelector('#type-facets');
    const tagFacets = document.querySelector('#tag-facets');
    
    if (typeFacets && facets.doc_types) {
      typeFacets.innerHTML = Object.entries(facets.doc_types)
        .map(([type, count]) => `
          <label class="facet-option">
            <input type="checkbox" value="${type}" onchange="this.filterSearchResults()">
            ${type} (${count})
          </label>
        `).join('');
    }
    
    if (tagFacets && facets.tags) {
      const topTags = Object.entries(facets.tags)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 10);
      
      tagFacets.innerHTML = topTags
        .map(([tag, count]) => `
          <label class="facet-option">
            <input type="checkbox" value="${tag}" onchange="this.filterSearchResults()">
            ${tag} (${count})
          </label>
        `).join('');
    }
  }
  
  async showSearchSuggestions(query) {
    const searchSuggestions = document.querySelector('#search-suggestions');
    
    if (!searchSuggestions) return;
    
    // Get suggestions (in production, this would call an API)
    const suggestions = this.generateSuggestions(query);
    
    if (suggestions.length > 0) {
      searchSuggestions.innerHTML = `
        <div class="suggestions-list">
          ${suggestions.map(suggestion => `
            <div class="suggestion-item" onclick="document.getElementById('doc-search').value='${suggestion}'; this.performAdvancedSearch('${suggestion}');">
              ${suggestion}
            </div>
          `).join('')}
        </div>
      `;
      searchSuggestions.style.display = 'block';
    } else {
      searchSuggestions.style.display = 'none';
    }
  }
  
  navigateSearchResults(direction) {
    const results = document.querySelectorAll('.search-result-item');
    const current = document.querySelector('.search-result-item.selected');
    
    if (results.length === 0) return;
    
    let newIndex = 0;
    
    if (current) {
      current.classList.remove('selected');
      const currentIndex = parseInt(current.dataset.index);
      
      if (direction === 'down') {
        newIndex = (currentIndex + 1) % results.length;
      } else {
        newIndex = currentIndex === 0 ? results.length - 1 : currentIndex - 1;
      }
    }
    
    results[newIndex].classList.add('selected');
    results[newIndex].scrollIntoView({ block: 'nearest' });
  }
  
  selectSearchResult() {
    const selected = document.querySelector('.search-result-item.selected');
    
    if (selected) {
      const link = selected.querySelector('a');
      if (link) {
        window.location.href = link.href;
      }
    }
  }
  
  filterSearchResults() {
    // Implement faceted filtering
    const typeFilters = Array.from(document.querySelectorAll('#type-facets input:checked')).map(cb => cb.value);
    const tagFilters = Array.from(document.querySelectorAll('#tag-facets input:checked')).map(cb => cb.value);
    
    const resultItems = document.querySelectorAll('.search-result-item');
    
    resultItems.forEach(item => {
      const resultData = this.getResultData(item);
      let show = true;
      
      if (typeFilters.length > 0 && !typeFilters.includes(resultData.doc_type)) {
        show = false;
      }
      
      if (tagFilters.length > 0 && !tagFilters.some(tag => resultData.tags.includes(tag))) {
        show = false;
      }
      
      item.style.display = show ? 'block' : 'none';
    });
  }
  
  getResultData(resultElement) {
    // Extract result data from DOM element
    // In production, this would be stored as data attributes
    return {
      doc_type: resultElement.querySelector('.search-result-type')?.textContent || 'general',
      tags: [] // Would be populated from data attributes
    };
  }
  
  clearSearch() {
    const searchInput = document.querySelector('#doc-search');
    const searchResults = document.querySelector('#search-results');
    const searchSuggestions = document.querySelector('#search-suggestions');
    
    if (searchInput) searchInput.value = '';
    if (searchResults) searchResults.style.display = 'none';
    if (searchSuggestions) searchSuggestions.style.display = 'none';
  }

  // ===== TABLE OF CONTENTS =====
  
  setupTableOfContents() {
    const tocContainer = document.querySelector('.toc-list');
    if (!tocContainer) return;
    
    const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
    const tocItems = [];
    
    headings.forEach((heading, index) => {
      // Generate ID if not present
      if (!heading.id) {
        heading.id = this.generateId(heading.textContent);
      }
      
      const level = parseInt(heading.tagName.charAt(1));
      tocItems.push({
        id: heading.id,
        text: heading.textContent,
        level: level,
        element: heading
      });
    });
    
    // Build TOC HTML
    const tocHTML = this.buildTocHTML(tocItems);
    tocContainer.innerHTML = tocHTML;
    
    // Add click handlers
    tocContainer.querySelectorAll('.toc-link').forEach(link => {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        const targetId = link.getAttribute('href').substring(1);
        const targetElement = document.getElementById(targetId);
        if (targetElement) {
          targetElement.scrollIntoView({ behavior: 'smooth' });
          
          // Close mobile nav if open
          if (window.innerWidth <= this.mobileBreakpoint) {
            this.closeMobileNav();
          }
        }
      });
    });
  }

  buildTocHTML(items) {
    return items.map(item => `
      <li class="toc-item toc-level-${item.level}">
        <a href="#${item.id}" class="toc-link">${item.text}</a>
      </li>
    `).join('');
  }

  generateId(text) {
    return text
      .toLowerCase()
      .replace(/[^\w\s-]/g, '')
      .replace(/\s+/g, '-')
      .trim();
  }

  // ===== SCROLL SPY =====
  
  setupScrollSpy() {
    const tocLinks = document.querySelectorAll('.toc-link');
    if (tocLinks.length === 0) return;
    
    const headings = Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6'))
      .filter(h => h.id);
    
    if (headings.length === 0) return;
    
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        const id = entry.target.id;
        const tocLink = document.querySelector(`.toc-link[href="#${id}"]`);
        
        if (entry.isIntersecting) {
          // Remove active class from all links
          tocLinks.forEach(link => link.classList.remove('active'));
          // Add active class to current link
          if (tocLink) tocLink.classList.add('active');
        }
      });
    }, {
      rootMargin: '-20% 0px -80% 0px'
    });
    
    headings.forEach(heading => observer.observe(heading));
  }

  // ===== ACCESSIBILITY =====
  
  setupAccessibility() {
    // Add skip link if not present
    if (!document.querySelector('.skip-link')) {
      const skipLink = document.createElement('a');
      skipLink.href = '#main-content';
      skipLink.className = 'skip-link';
      skipLink.textContent = 'Skip to main content';
      document.body.insertBefore(skipLink, document.body.firstChild);
    }
    
    // Ensure main content has proper ID
    const mainContent = document.querySelector('.doc-main');
    if (mainContent && !mainContent.id) {
      mainContent.id = 'main-content';
    }
    
    // Add ARIA labels to interactive elements
    this.addAriaLabels();
    
    // Setup keyboard navigation
    this.setupKeyboardNavigation();
  }

  addAriaLabels() {
    // Theme toggle
    const themeToggle = document.querySelector('.theme-toggle');
    if (themeToggle && !themeToggle.getAttribute('aria-label')) {
      themeToggle.setAttribute('aria-label', `Switch to ${this.currentTheme === 'dark' ? 'light' : 'dark'} theme`);
    }
    
    // Mobile nav toggle
    const mobileToggle = document.querySelector('.mobile-nav-toggle');
    if (mobileToggle && !mobileToggle.getAttribute('aria-label')) {
      mobileToggle.setAttribute('aria-label', 'Toggle navigation menu');
      mobileToggle.setAttribute('aria-expanded', 'false');
    }
    
    // Search input
    const searchInput = document.querySelector('.search-input');
    if (searchInput && !searchInput.getAttribute('aria-label')) {
      searchInput.setAttribute('aria-label', 'Search documentation');
    }
  }

  setupKeyboardNavigation() {
    // Tab trap for mobile navigation
    const mobileSidebar = document.querySelector('.mobile-sidebar');
    if (mobileSidebar) {
      mobileSidebar.addEventListener('keydown', (e) => {
        if (e.key === 'Tab') {
          this.trapFocus(e, mobileSidebar);
        }
      });
    }
  }

  trapFocus(e, container) {
    const focusableElements = container.querySelectorAll(
      'a, button, input, textarea, select, [tabindex]:not([tabindex="-1"])'
    );
    
    const firstFocusable = focusableElements[0];
    const lastFocusable = focusableElements[focusableElements.length - 1];
    
    if (e.shiftKey) {
      if (document.activeElement === firstFocusable) {
        e.preventDefault();
        lastFocusable.focus();
      }
    } else {
      if (document.activeElement === lastFocusable) {
        e.preventDefault();
        firstFocusable.focus();
      }
    }
  }

  // ===== RESPONSIVE IMAGES =====
  
  setupResponsiveImages() {
    const images = document.querySelectorAll('img');
    
    images.forEach(img => {
      // Add loading="lazy" if not present
      if (!img.getAttribute('loading')) {
        img.setAttribute('loading', 'lazy');
      }
      
      // Wrap images in responsive containers
      if (!img.parentElement.classList.contains('img-responsive')) {
        const wrapper = document.createElement('div');
        wrapper.className = 'img-responsive';
        img.parentNode.insertBefore(wrapper, img);
        wrapper.appendChild(img);
      }
    });
  }

  // ===== CODE COPY BUTTONS =====
  
  setupCodeCopyButtons() {
    const codeBlocks = document.querySelectorAll('pre code');
    
    codeBlocks.forEach(codeBlock => {
      const pre = codeBlock.parentElement;
      
      // Skip if copy button already exists
      if (pre.querySelector('.copy-button')) return;
      
      const copyButton = document.createElement('button');
      copyButton.className = 'copy-button';
      copyButton.innerHTML = '📋';
      copyButton.setAttribute('aria-label', 'Copy code to clipboard');
      copyButton.setAttribute('title', 'Copy code');
      
      copyButton.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(codeBlock.textContent);
          copyButton.innerHTML = '✅';
          copyButton.setAttribute('title', 'Copied!');
          
          setTimeout(() => {
            copyButton.innerHTML = '📋';
            copyButton.setAttribute('title', 'Copy code');
          }, 2000);
        } catch (err) {
          console.error('Failed to copy code:', err);
          copyButton.innerHTML = '❌';
          setTimeout(() => {
            copyButton.innerHTML = '📋';
          }, 2000);
        }
      });
      
      pre.style.position = 'relative';
      pre.appendChild(copyButton);
    });
  }

  // ===== RESPONSIVE HANDLING =====
  
  handleResize() {
    const width = window.innerWidth;
    
    // Close mobile nav on resize to desktop
    if (width > this.mobileBreakpoint) {
      this.closeMobileNav();
    }
    
    // Update responsive classes
    document.body.classList.toggle('mobile', width <= this.mobileBreakpoint);
    document.body.classList.toggle('tablet', width > this.mobileBreakpoint && width <= this.tabletBreakpoint);
    document.body.classList.toggle('desktop', width > this.tabletBreakpoint);
  }

  // ===== UTILITIES =====
  
  debounce(func, wait) {
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
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new DocumentationFramework();
});

// CSS for copy buttons and responsive images
const additionalStyles = `
<style>
.copy-button {
  position: absolute;
  top: 0.5rem;
  right: 0.5rem;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-sm);
  padding: 0.25rem 0.5rem;
  font-size: 0.875rem;
  cursor: pointer;
  opacity: 0;
  transition: opacity var(--transition-fast);
}

pre:hover .copy-button {
  opacity: 1;
}

.copy-button:hover {
  background: var(--bg-tertiary);
}

.img-responsive {
  max-width: 100%;
  height: auto;
  margin: var(--spacing-md) 0;
  text-align: center;
}

.img-responsive img {
  max-width: 100%;
  height: auto;
  border-radius: var(--border-radius-md);
  box-shadow: var(--shadow-sm);
}

.toc-level-1 { margin-left: 0; }
.toc-level-2 { margin-left: 1rem; }
.toc-level-3 { margin-left: 2rem; }
.toc-level-4 { margin-left: 3rem; }
.toc-level-5 { margin-left: 4rem; }
.toc-level-6 { margin-left: 5rem; }

.search-result-title {
  font-weight: 600;
  margin-bottom: 0.25rem;
}

.search-result-snippet {
  font-size: var(--font-size-sm);
  color: var(--text-secondary);
}

.search-result-snippet mark {
  background-color: var(--bg-highlight);
  padding: 0.1em 0.2em;
  border-radius: var(--border-radius-sm);
}

@media (max-width: 768px) {
  .mobile-nav-toggle {
    display: block;
  }
  
  .doc-sidebar {
    display: none;
  }
}

@media (min-width: 769px) {
  .mobile-nav-toggle,
  .mobile-sidebar,
  .mobile-overlay {
    display: none;
  }
}
</style>
`;

// Inject additional styles
document.head.insertAdjacentHTML('beforeend', additionalStyles);