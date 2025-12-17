#!/usr/bin/env python3
"""
Intelligent Navigation Engine for Documentation System

This module implements advanced navigation features including automatic cross-referencing,
breadcrumb navigation, contextual linking, and "related content" suggestions based on
content analysis and semantic relationships.

Implements Requirements 7.1, 8.2 from the comprehensive documentation specification.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, Counter
from urllib.parse import urljoin, urlparse
import hashlib

# Import search engine for content analysis
try:
    from .search_engine import DocumentationSearchEngine, SearchDocument, DocumentType
except ImportError:
    from search_engine import DocumentationSearchEngine, SearchDocument, DocumentType


@dataclass
class NavigationNode:
    """Represents a node in the navigation hierarchy."""
    id: str
    title: str
    url: str
    path: str
    doc_type: DocumentType
    level: int = 0
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    last_modified: datetime = field(default_factory=datetime.now)


@dataclass
class CrossReference:
    """Represents a cross-reference between documents."""
    source_id: str
    target_id: str
    reference_type: str  # 'link', 'mention', 'related', 'dependency'
    context: str
    confidence: float
    anchor_text: str = ""
    line_number: Optional[int] = None


@dataclass
class BreadcrumbItem:
    """Represents an item in a breadcrumb trail."""
    title: str
    url: str
    is_current: bool = False


@dataclass
class RelatedContent:
    """Represents related content suggestions."""
    document: SearchDocument
    relationship_type: str  # 'similar', 'referenced', 'category', 'tag'
    relevance_score: float
    explanation: str


@dataclass
class NavigationContext:
    """Represents the navigation context for a document."""
    current_document: SearchDocument
    breadcrumbs: List[BreadcrumbItem]
    cross_references: List[CrossReference]
    related_content: List[RelatedContent]
    navigation_tree: Dict[str, NavigationNode]
    prev_document: Optional[SearchDocument] = None
    next_document: Optional[SearchDocument] = None


class ContentAnalyzer:
    """Analyzes content for semantic relationships and cross-references."""
    
    def __init__(self):
        self.link_patterns = [
            # Markdown links
            r'\[([^\]]+)\]\(([^)]+)\)',
            # HTML links
            r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>',
            # Reference-style links
            r'\[([^\]]+)\]\[([^\]]+)\]',
            # Wiki-style links
            r'\[\[([^\]]+)\]\]'
        ]
        
        self.mention_patterns = [
            # API endpoints
            r'`(GET|POST|PUT|DELETE|PATCH)\s+([^`]+)`',
            # Code references
            r'`([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)`',
            # File references
            r'`([^`]+\.(py|js|ts|tsx|md|json|yaml|yml))`',
            # Class/function references
            r'(?:class|function|method)\s+`([^`]+)`'
        ]
        
        # Common technical terms for relationship detection
        self.technical_terms = {
            'api', 'endpoint', 'service', 'component', 'model', 'schema',
            'database', 'table', 'query', 'function', 'method', 'class',
            'interface', 'type', 'configuration', 'deployment', 'workflow',
            'algorithm', 'architecture', 'system', 'module', 'library'
        }
    
    def extract_links(self, content: str, base_url: str = "") -> List[Dict[str, Any]]:
        """Extract all links from content."""
        links = []
        
        for pattern in self.link_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                if pattern.startswith(r'\['):  # Markdown link
                    anchor_text = match.group(1)
                    url = match.group(2)
                elif pattern.startswith(r'<a'):  # HTML link
                    url = match.group(1)
                    anchor_text = match.group(2)
                else:
                    anchor_text = match.group(1)
                    url = match.group(2) if match.lastindex > 1 else match.group(1)
                
                # Calculate line number
                line_number = content[:match.start()].count('\n') + 1
                
                # Resolve relative URLs
                if base_url and not url.startswith(('http://', 'https://', '#')):
                    url = urljoin(base_url, url)
                
                links.append({
                    'anchor_text': anchor_text.strip(),
                    'url': url.strip(),
                    'line_number': line_number,
                    'context': self._extract_context(content, match.start(), match.end())
                })
        
        return links
    
    def extract_mentions(self, content: str) -> List[Dict[str, Any]]:
        """Extract mentions of technical terms and references."""
        mentions = []
        
        for pattern in self.mention_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                mention_text = match.group(1) if match.lastindex >= 1 else match.group(0)
                line_number = content[:match.start()].count('\n') + 1
                
                mentions.append({
                    'text': mention_text.strip(),
                    'type': self._classify_mention(mention_text),
                    'line_number': line_number,
                    'context': self._extract_context(content, match.start(), match.end())
                })
        
        return mentions
    
    def _extract_context(self, content: str, start: int, end: int, 
                        context_length: int = 100) -> str:
        """Extract context around a match."""
        context_start = max(0, start - context_length)
        context_end = min(len(content), end + context_length)
        
        context = content[context_start:context_end]
        
        # Clean up context
        context = re.sub(r'\s+', ' ', context).strip()
        
        # Add ellipsis if truncated
        if context_start > 0:
            context = "..." + context
        if context_end < len(content):
            context = context + "..."
        
        return context
    
    def _classify_mention(self, mention_text: str) -> str:
        """Classify the type of mention."""
        mention_lower = mention_text.lower()
        
        if any(term in mention_lower for term in ['get', 'post', 'put', 'delete', 'patch']):
            return 'api_endpoint'
        elif mention_text.endswith(('.py', '.js', '.ts', '.tsx', '.md', '.json', '.yaml', '.yml')):
            return 'file_reference'
        elif '.' in mention_text and not mention_text.startswith('.'):
            return 'code_reference'
        elif any(term in mention_lower for term in self.technical_terms):
            return 'technical_term'
        else:
            return 'general_reference'
    
    def calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two pieces of content."""
        # Simple word-based similarity
        words1 = set(re.findall(r'\b\w+\b', content1.lower()))
        words2 = set(re.findall(r'\b\w+\b', content2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        # Jaccard similarity
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Boost similarity for technical terms
        technical_intersection = intersection & self.technical_terms
        if technical_intersection:
            jaccard *= (1 + len(technical_intersection) * 0.1)
        
        return min(jaccard, 1.0)
    
    def extract_headings(self, content: str) -> List[Dict[str, Any]]:
        """Extract headings for navigation structure."""
        headings = []
        
        # Markdown headings
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        matches = re.finditer(heading_pattern, content, re.MULTILINE)
        
        for match in matches:
            level = len(match.group(1))
            title = match.group(2).strip()
            line_number = content[:match.start()].count('\n') + 1
            
            # Generate anchor ID
            anchor_id = re.sub(r'[^\w\s-]', '', title.lower())
            anchor_id = re.sub(r'[-\s]+', '-', anchor_id).strip('-')
            
            headings.append({
                'level': level,
                'title': title,
                'anchor_id': anchor_id,
                'line_number': line_number
            })
        
        return headings


class NavigationEngine:
    """
    Intelligent navigation engine for documentation system.
    
    Provides automatic cross-referencing, breadcrumb navigation, contextual linking,
    and related content suggestions based on content analysis and semantic relationships.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.docs_dir = project_root / "docs"
        self.cache_dir = project_root / ".kiro" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize content analyzer
        self.content_analyzer = ContentAnalyzer()
        
        # Initialize search engine for content analysis
        self.search_engine = DocumentationSearchEngine(project_root)
        
        # Navigation data
        self.navigation_tree: Dict[str, NavigationNode] = {}
        self.cross_references: List[CrossReference] = []
        self.document_index: Dict[str, SearchDocument] = {}
        
        # Cache files
        self.nav_cache_file = self.cache_dir / "navigation_cache.json"
        self.xref_cache_file = self.cache_dir / "cross_references.json"
        
        # Load cached data
        self._load_cached_data()
    
    def build_navigation_structure(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """Build the complete navigation structure from documentation."""
        print("🧭 Building navigation structure...")
        
        if not force_rebuild and self._is_cache_valid():
            print("  📋 Using cached navigation structure")
            return self._get_navigation_summary()
        
        # Clear existing data
        self.navigation_tree.clear()
        self.cross_references.clear()
        self.document_index.clear()
        
        # Index all documents first
        self._index_documents()
        
        # Build navigation hierarchy
        self._build_hierarchy()
        
        # Extract cross-references
        self._extract_cross_references()
        
        # Calculate related content relationships
        self._calculate_relationships()
        
        # Save to cache
        self._save_cached_data()
        
        summary = self._get_navigation_summary()
        print(f"  ✅ Built navigation for {summary['total_documents']} documents")
        print(f"  🔗 Found {summary['cross_references']} cross-references")
        
        return summary
    
    def _index_documents(self):
        """Index all documentation files."""
        for md_file in self.docs_dir.rglob("*.md"):
            try:
                # Skip certain files
                if any(skip in str(md_file) for skip in ['.git', '__pycache__', 'node_modules']):
                    continue
                
                document = self._create_document_from_file(md_file)
                if document:
                    self.document_index[document.id] = document
                    
                    # Create navigation node
                    nav_node = self._create_navigation_node(document, md_file)
                    self.navigation_tree[document.id] = nav_node
                    
            except Exception as e:
                print(f"Error indexing {md_file}: {e}")
    
    def _create_document_from_file(self, file_path: Path) -> Optional[SearchDocument]:
        """Create a SearchDocument from a file."""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Extract title
            title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            title = title_match.group(1) if title_match else file_path.stem.replace('-', ' ').title()
            
            # Determine document type
            doc_type = self._determine_document_type(file_path)
            
            # Extract tags
            tags = self._extract_tags_from_path(file_path)
            
            # Create relative URL
            relative_path = file_path.relative_to(self.docs_dir)
            url = f"/docs/{relative_path.as_posix()}"
            
            # Metadata
            metadata = {
                'file_size': file_path.stat().st_size,
                'word_count': len(content.split()),
                'headings': self.content_analyzer.extract_headings(content)
            }
            
            last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
            
            return SearchDocument(
                id=str(file_path.relative_to(self.project_root)),
                title=title,
                content=content,
                doc_type=doc_type,
                file_path=str(file_path.relative_to(self.project_root)),
                url=url,
                tags=tags,
                metadata=metadata,
                last_modified=last_modified
            )
            
        except Exception as e:
            print(f"Error creating document from {file_path}: {e}")
            return None
    
    def _create_navigation_node(self, document: SearchDocument, file_path: Path) -> NavigationNode:
        """Create a navigation node from a document."""
        # Determine hierarchy level from path depth
        relative_path = file_path.relative_to(self.docs_dir)
        level = len(relative_path.parts) - 1
        
        # Determine parent from path
        parent_id = None
        if level > 0:
            parent_path = relative_path.parent
            parent_file = self.docs_dir / parent_path / "README.md"
            if parent_file.exists():
                parent_id = str(parent_file.relative_to(self.project_root))
            else:
                # Look for index file
                index_file = self.docs_dir / parent_path / "index.md"
                if index_file.exists():
                    parent_id = str(index_file.relative_to(self.project_root))
        
        return NavigationNode(
            id=document.id,
            title=document.title,
            url=document.url,
            path=document.file_path,
            doc_type=document.doc_type,
            level=level,
            parent_id=parent_id,
            metadata=document.metadata,
            tags=document.tags,
            last_modified=document.last_modified
        )
    
    def _build_hierarchy(self):
        """Build parent-child relationships in navigation tree."""
        # First pass: establish parent-child relationships
        for node_id, node in self.navigation_tree.items():
            if node.parent_id and node.parent_id in self.navigation_tree:
                parent_node = self.navigation_tree[node.parent_id]
                if node_id not in parent_node.children:
                    parent_node.children.append(node_id)
        
        # Second pass: sort children by title
        for node in self.navigation_tree.values():
            node.children.sort(key=lambda child_id: self.navigation_tree[child_id].title)
    
    def _extract_cross_references(self):
        """Extract cross-references between documents."""
        for doc_id, document in self.document_index.items():
            # Extract links
            links = self.content_analyzer.extract_links(document.content, document.url)
            
            for link in links:
                target_id = self._resolve_link_target(link['url'])
                if target_id and target_id in self.document_index:
                    cross_ref = CrossReference(
                        source_id=doc_id,
                        target_id=target_id,
                        reference_type='link',
                        context=link['context'],
                        confidence=1.0,
                        anchor_text=link['anchor_text'],
                        line_number=link['line_number']
                    )
                    self.cross_references.append(cross_ref)
            
            # Extract mentions
            mentions = self.content_analyzer.extract_mentions(document.content)
            
            for mention in mentions:
                # Try to find documents that match the mention
                matching_docs = self._find_documents_by_mention(mention['text'])
                
                for target_id in matching_docs:
                    if target_id != doc_id:
                        cross_ref = CrossReference(
                            source_id=doc_id,
                            target_id=target_id,
                            reference_type='mention',
                            context=mention['context'],
                            confidence=0.7,
                            anchor_text=mention['text'],
                            line_number=mention['line_number']
                        )
                        self.cross_references.append(cross_ref)
    
    def _resolve_link_target(self, url: str) -> Optional[str]:
        """Resolve a URL to a document ID."""
        # Handle relative URLs
        if url.startswith('/docs/'):
            # Convert URL back to file path
            relative_url = url[6:]  # Remove '/docs/'
            file_path = self.docs_dir / relative_url
            
            # Handle .html extensions (convert to .md)
            if file_path.suffix == '.html':
                file_path = file_path.with_suffix('.md')
            
            if file_path.exists():
                return str(file_path.relative_to(self.project_root))
        
        # Handle anchor links within the same document
        if url.startswith('#'):
            return None  # Internal anchor, not a cross-reference
        
        # Handle external links
        if url.startswith(('http://', 'https://')):
            return None  # External link, not tracked
        
        return None
    
    def _find_documents_by_mention(self, mention_text: str) -> List[str]:
        """Find documents that might be referenced by a mention."""
        matching_docs = []
        mention_lower = mention_text.lower()
        
        for doc_id, document in self.document_index.items():
            # Check if mention appears in title
            if mention_lower in document.title.lower():
                matching_docs.append(doc_id)
                continue
            
            # Check if mention appears in file name
            file_name = Path(document.file_path).stem.lower()
            if mention_lower in file_name or file_name in mention_lower:
                matching_docs.append(doc_id)
                continue
            
            # Check tags
            if any(mention_lower in tag.lower() for tag in document.tags):
                matching_docs.append(doc_id)
        
        return matching_docs
    
    def _calculate_relationships(self):
        """Calculate semantic relationships between documents."""
        # This is a simplified implementation
        # In a production system, you might use more sophisticated NLP techniques
        
        for doc_id, document in self.document_index.items():
            # Find similar documents based on content
            for other_id, other_doc in self.document_index.items():
                if doc_id != other_id:
                    similarity = self.content_analyzer.calculate_content_similarity(
                        document.content, other_doc.content
                    )
                    
                    if similarity > 0.3:  # Threshold for relatedness
                        cross_ref = CrossReference(
                            source_id=doc_id,
                            target_id=other_id,
                            reference_type='related',
                            context=f"Content similarity: {similarity:.2f}",
                            confidence=similarity,
                            anchor_text=""
                        )
                        self.cross_references.append(cross_ref)
    
    def get_navigation_context(self, document_id: str) -> NavigationContext:
        """Get complete navigation context for a document."""
        if document_id not in self.document_index:
            raise ValueError(f"Document {document_id} not found")
        
        current_document = self.document_index[document_id]
        
        # Generate breadcrumbs
        breadcrumbs = self._generate_breadcrumbs(document_id)
        
        # Get cross-references for this document
        doc_cross_refs = [
            xref for xref in self.cross_references 
            if xref.source_id == document_id or xref.target_id == document_id
        ]
        
        # Get related content
        related_content = self._get_related_content(document_id)
        
        # Get previous/next documents in sequence
        prev_doc, next_doc = self._get_sequential_navigation(document_id)
        
        return NavigationContext(
            current_document=current_document,
            breadcrumbs=breadcrumbs,
            cross_references=doc_cross_refs,
            related_content=related_content,
            navigation_tree=self.navigation_tree,
            prev_document=prev_doc,
            next_document=next_doc
        )
    
    def _generate_breadcrumbs(self, document_id: str) -> List[BreadcrumbItem]:
        """Generate breadcrumb trail for a document."""
        breadcrumbs = []
        
        if document_id not in self.navigation_tree:
            return breadcrumbs
        
        # Build path from root to current document
        path = []
        current_id = document_id
        
        while current_id:
            node = self.navigation_tree[current_id]
            path.append(node)
            current_id = node.parent_id
        
        # Reverse to get root-to-current order
        path.reverse()
        
        # Convert to breadcrumb items
        for i, node in enumerate(path):
            breadcrumbs.append(BreadcrumbItem(
                title=node.title,
                url=node.url,
                is_current=(i == len(path) - 1)
            ))
        
        return breadcrumbs
    
    def _get_related_content(self, document_id: str, limit: int = 10) -> List[RelatedContent]:
        """Get related content suggestions for a document."""
        related = []
        current_doc = self.document_index[document_id]
        
        # Get documents referenced by this document
        outgoing_refs = [
            xref for xref in self.cross_references 
            if xref.source_id == document_id and xref.confidence > 0.5
        ]
        
        for xref in outgoing_refs[:5]:  # Limit to top 5
            if xref.target_id in self.document_index:
                target_doc = self.document_index[xref.target_id]
                related.append(RelatedContent(
                    document=target_doc,
                    relationship_type='referenced',
                    relevance_score=xref.confidence,
                    explanation=f"Referenced in current document: {xref.anchor_text}"
                ))
        
        # Get documents that reference this document
        incoming_refs = [
            xref for xref in self.cross_references 
            if xref.target_id == document_id and xref.confidence > 0.5
        ]
        
        for xref in incoming_refs[:3]:  # Limit to top 3
            if xref.source_id in self.document_index:
                source_doc = self.document_index[xref.source_id]
                related.append(RelatedContent(
                    document=source_doc,
                    relationship_type='references_this',
                    relevance_score=xref.confidence,
                    explanation=f"References this document: {xref.anchor_text}"
                ))
        
        # Get documents with similar tags
        for doc_id, document in self.document_index.items():
            if doc_id != document_id:
                common_tags = set(current_doc.tags) & set(document.tags)
                if common_tags:
                    score = len(common_tags) / max(len(current_doc.tags), len(document.tags), 1)
                    if score > 0.3:
                        related.append(RelatedContent(
                            document=document,
                            relationship_type='similar_tags',
                            relevance_score=score,
                            explanation=f"Shares tags: {', '.join(common_tags)}"
                        ))
        
        # Get documents of the same type
        same_type_docs = [
            doc for doc_id, doc in self.document_index.items()
            if doc_id != document_id and doc.doc_type == current_doc.doc_type
        ]
        
        for doc in same_type_docs[:3]:  # Limit to top 3
            related.append(RelatedContent(
                document=doc,
                relationship_type='same_category',
                relevance_score=0.4,
                explanation=f"Same category: {doc.doc_type.value}"
            ))
        
        # Sort by relevance score and limit results
        related.sort(key=lambda x: x.relevance_score, reverse=True)
        return related[:limit]
    
    def _get_sequential_navigation(self, document_id: str) -> Tuple[Optional[SearchDocument], Optional[SearchDocument]]:
        """Get previous and next documents in logical sequence."""
        if document_id not in self.navigation_tree:
            return None, None
        
        current_node = self.navigation_tree[document_id]
        
        # If document has a parent, get siblings for sequential navigation
        if current_node.parent_id and current_node.parent_id in self.navigation_tree:
            parent_node = self.navigation_tree[current_node.parent_id]
            siblings = parent_node.children
            
            try:
                current_index = siblings.index(document_id)
                
                prev_doc = None
                if current_index > 0:
                    prev_id = siblings[current_index - 1]
                    prev_doc = self.document_index.get(prev_id)
                
                next_doc = None
                if current_index < len(siblings) - 1:
                    next_id = siblings[current_index + 1]
                    next_doc = self.document_index.get(next_id)
                
                return prev_doc, next_doc
                
            except ValueError:
                pass
        
        return None, None
    
    def generate_sitemap(self) -> Dict[str, Any]:
        """Generate a complete sitemap of the documentation."""
        sitemap = {
            'title': 'Documentation Sitemap',
            'generated_at': datetime.now().isoformat(),
            'total_documents': len(self.document_index),
            'sections': {}
        }
        
        # Group documents by type
        by_type = defaultdict(list)
        for doc_id, document in self.document_index.items():
            by_type[document.doc_type.value].append({
                'id': doc_id,
                'title': document.title,
                'url': document.url,
                'last_modified': document.last_modified.isoformat()
            })
        
        # Sort documents within each type
        for doc_type, docs in by_type.items():
            docs.sort(key=lambda x: x['title'])
            sitemap['sections'][doc_type] = docs
        
        return sitemap
    
    def generate_cross_reference_report(self) -> Dict[str, Any]:
        """Generate a report of all cross-references."""
        report = {
            'title': 'Cross-Reference Report',
            'generated_at': datetime.now().isoformat(),
            'total_references': len(self.cross_references),
            'by_type': defaultdict(int),
            'broken_links': [],
            'most_referenced': [],
            'orphaned_documents': []
        }
        
        # Count by type
        for xref in self.cross_references:
            report['by_type'][xref.reference_type] += 1
        
        # Find broken links
        for xref in self.cross_references:
            if xref.target_id not in self.document_index:
                report['broken_links'].append({
                    'source': self.document_index[xref.source_id].title,
                    'target_id': xref.target_id,
                    'anchor_text': xref.anchor_text,
                    'context': xref.context[:100] + "..." if len(xref.context) > 100 else xref.context
                })
        
        # Find most referenced documents
        reference_counts = Counter(xref.target_id for xref in self.cross_references)
        for doc_id, count in reference_counts.most_common(10):
            if doc_id in self.document_index:
                report['most_referenced'].append({
                    'title': self.document_index[doc_id].title,
                    'url': self.document_index[doc_id].url,
                    'reference_count': count
                })
        
        # Find orphaned documents (no incoming references)
        referenced_docs = set(xref.target_id for xref in self.cross_references)
        for doc_id, document in self.document_index.items():
            if doc_id not in referenced_docs:
                report['orphaned_documents'].append({
                    'title': document.title,
                    'url': document.url,
                    'doc_type': document.doc_type.value
                })
        
        return report
    
    def _determine_document_type(self, file_path: Path) -> DocumentType:
        """Determine document type from file path."""
        path_str = str(file_path).lower()
        
        type_mapping = {
            'api': DocumentType.API,
            'architecture': DocumentType.ARCHITECTURE,
            'algorithms': DocumentType.ALGORITHM,
            'workflows': DocumentType.WORKFLOW,
            'interfaces': DocumentType.INTERFACE,
            'deployment': DocumentType.DEPLOYMENT,
            'frontend': DocumentType.FRONTEND,
            'database': DocumentType.DATABASE
        }
        
        for key, doc_type in type_mapping.items():
            if key in path_str:
                return doc_type
        
        return DocumentType.GENERAL
    
    def _extract_tags_from_path(self, file_path: Path) -> List[str]:
        """Extract tags from file path."""
        tags = []
        
        # Add tags based on path components
        for part in file_path.parts:
            if part.lower() in ['api', 'architecture', 'algorithms', 'workflows', 
                               'interfaces', 'deployment', 'frontend', 'database']:
                tags.append(part.lower())
        
        # Add tag based on filename
        stem = file_path.stem.lower()
        if stem not in ['readme', 'index']:
            tags.append(stem.replace('-', '_'))
        
        return list(set(tags))
    
    def _is_cache_valid(self) -> bool:
        """Check if cached navigation data is still valid."""
        if not self.nav_cache_file.exists() or not self.xref_cache_file.exists():
            return False
        
        # Check if any documentation files are newer than cache
        cache_time = self.nav_cache_file.stat().st_mtime
        
        for md_file in self.docs_dir.rglob("*.md"):
            if md_file.stat().st_mtime > cache_time:
                return False
        
        return True
    
    def _load_cached_data(self):
        """Load navigation data from cache."""
        try:
            if self.nav_cache_file.exists():
                with open(self.nav_cache_file, 'r') as f:
                    nav_data = json.load(f)
                
                # Reconstruct navigation tree
                for node_data in nav_data.get('navigation_tree', []):
                    node = NavigationNode(
                        id=node_data['id'],
                        title=node_data['title'],
                        url=node_data['url'],
                        path=node_data['path'],
                        doc_type=DocumentType(node_data['doc_type']),
                        level=node_data['level'],
                        parent_id=node_data.get('parent_id'),
                        children=node_data.get('children', []),
                        metadata=node_data.get('metadata', {}),
                        tags=node_data.get('tags', []),
                        last_modified=datetime.fromisoformat(node_data['last_modified'])
                    )
                    self.navigation_tree[node.id] = node
                
                # Reconstruct document index
                for doc_data in nav_data.get('document_index', []):
                    doc = SearchDocument(
                        id=doc_data['id'],
                        title=doc_data['title'],
                        content=doc_data['content'],
                        doc_type=DocumentType(doc_data['doc_type']),
                        file_path=doc_data['file_path'],
                        url=doc_data['url'],
                        tags=doc_data.get('tags', []),
                        metadata=doc_data.get('metadata', {}),
                        last_modified=datetime.fromisoformat(doc_data['last_modified'])
                    )
                    self.document_index[doc.id] = doc
            
            if self.xref_cache_file.exists():
                with open(self.xref_cache_file, 'r') as f:
                    xref_data = json.load(f)
                
                # Reconstruct cross-references
                for xref_data_item in xref_data.get('cross_references', []):
                    xref = CrossReference(
                        source_id=xref_data_item['source_id'],
                        target_id=xref_data_item['target_id'],
                        reference_type=xref_data_item['reference_type'],
                        context=xref_data_item['context'],
                        confidence=xref_data_item['confidence'],
                        anchor_text=xref_data_item.get('anchor_text', ''),
                        line_number=xref_data_item.get('line_number')
                    )
                    self.cross_references.append(xref)
                    
        except Exception as e:
            print(f"Error loading cached navigation data: {e}")
            # Clear corrupted cache
            self.navigation_tree.clear()
            self.cross_references.clear()
            self.document_index.clear()
    
    def _save_cached_data(self):
        """Save navigation data to cache."""
        try:
            # Save navigation tree and document index
            nav_data = {
                'navigation_tree': [
                    {
                        'id': node.id,
                        'title': node.title,
                        'url': node.url,
                        'path': node.path,
                        'doc_type': node.doc_type.value,
                        'level': node.level,
                        'parent_id': node.parent_id,
                        'children': node.children,
                        'metadata': node.metadata,
                        'tags': node.tags,
                        'last_modified': node.last_modified.isoformat()
                    }
                    for node in self.navigation_tree.values()
                ],
                'document_index': [
                    {
                        'id': doc.id,
                        'title': doc.title,
                        'content': doc.content,
                        'doc_type': doc.doc_type.value,
                        'file_path': doc.file_path,
                        'url': doc.url,
                        'tags': doc.tags,
                        'metadata': doc.metadata,
                        'last_modified': doc.last_modified.isoformat()
                    }
                    for doc in self.document_index.values()
                ]
            }
            
            with open(self.nav_cache_file, 'w') as f:
                json.dump(nav_data, f, indent=2)
            
            # Save cross-references
            xref_data = {
                'cross_references': [
                    {
                        'source_id': xref.source_id,
                        'target_id': xref.target_id,
                        'reference_type': xref.reference_type,
                        'context': xref.context,
                        'confidence': xref.confidence,
                        'anchor_text': xref.anchor_text,
                        'line_number': xref.line_number
                    }
                    for xref in self.cross_references
                ]
            }
            
            with open(self.xref_cache_file, 'w') as f:
                json.dump(xref_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving navigation cache: {e}")
    
    def _get_navigation_summary(self) -> Dict[str, Any]:
        """Get summary of navigation structure."""
        return {
            'total_documents': len(self.document_index),
            'navigation_nodes': len(self.navigation_tree),
            'cross_references': len(self.cross_references),
            'document_types': dict(Counter(doc.doc_type.value for doc in self.document_index.values())),
            'max_depth': max((node.level for node in self.navigation_tree.values()), default=0),
            'last_updated': max((doc.last_modified for doc in self.document_index.values()), default=datetime.now()).isoformat()
        }
    
    def close(self):
        """Close navigation engine and cleanup resources."""
        if hasattr(self, 'search_engine'):
            self.search_engine.close()


# CLI interface for testing and maintenance
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Documentation Navigation Engine")
    parser.add_argument("--build", action="store_true", help="Build navigation structure")
    parser.add_argument("--sitemap", action="store_true", help="Generate sitemap")
    parser.add_argument("--xref-report", action="store_true", help="Generate cross-reference report")
    parser.add_argument("--context", type=str, help="Get navigation context for document")
    parser.add_argument("--project-root", type=str, default=".", help="Project root directory")
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    nav_engine = NavigationEngine(project_root)
    
    try:
        if args.build:
            result = nav_engine.build_navigation_structure(force_rebuild=True)
            print(f"Navigation structure built: {result}")
        
        if args.sitemap:
            sitemap = nav_engine.generate_sitemap()
            print(json.dumps(sitemap, indent=2))
        
        if args.xref_report:
            report = nav_engine.generate_cross_reference_report()
            print(json.dumps(report, indent=2))
        
        if args.context:
            try:
                context = nav_engine.get_navigation_context(args.context)
                print(f"Navigation context for {context.current_document.title}:")
                print(f"  Breadcrumbs: {' > '.join(b.title for b in context.breadcrumbs)}")
                print(f"  Cross-references: {len(context.cross_references)}")
                print(f"  Related content: {len(context.related_content)}")
            except ValueError as e:
                print(f"Error: {e}")
    
    finally:
        nav_engine.close()