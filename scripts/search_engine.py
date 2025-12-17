#!/usr/bin/env python3
"""
Advanced Search Engine for Documentation System

This module implements a comprehensive search infrastructure with full-text indexing,
faceted filtering, relevance scoring, and intelligent query processing for all
documentation content.

Implements Requirements 7.1 from the comprehensive documentation specification.
"""

import os
import re
import json
import sqlite3
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math
from collections import defaultdict, Counter
import unicodedata

# Text processing imports
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class DocumentType(Enum):
    """Types of documentation content."""
    API = "api"
    ARCHITECTURE = "architecture"
    ALGORITHM = "algorithm"
    WORKFLOW = "workflow"
    INTERFACE = "interface"
    DEPLOYMENT = "deployment"
    FRONTEND = "frontend"
    DATABASE = "database"
    GENERAL = "general"


@dataclass
class SearchDocument:
    """Represents a searchable document in the index."""
    id: str
    title: str
    content: str
    doc_type: DocumentType
    file_path: str
    url: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_modified: datetime = field(default_factory=datetime.now)
    content_hash: str = ""
    
    def __post_init__(self):
        """Calculate content hash after initialization."""
        if not self.content_hash:
            self.content_hash = hashlib.md5(
                (self.title + self.content).encode('utf-8')
            ).hexdigest()


@dataclass
class SearchQuery:
    """Represents a search query with filters and options."""
    query: str
    doc_types: List[DocumentType] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = 50
    offset: int = 0
    include_content: bool = True
    highlight: bool = True


@dataclass
class SearchResult:
    """Represents a single search result."""
    document: SearchDocument
    score: float
    highlights: List[str] = field(default_factory=list)
    snippet: str = ""
    matched_terms: List[str] = field(default_factory=list)


@dataclass
class SearchResponse:
    """Represents the complete search response."""
    results: List[SearchResult]
    total_count: int
    query_time: float
    facets: Dict[str, Dict[str, int]] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    query: SearchQuery = None


class TextProcessor:
    """Advanced text processing for search indexing and querying."""
    
    def __init__(self):
        self.stemmer = None
        self.stop_words = set()
        
        if NLTK_AVAILABLE:
            try:
                # Download required NLTK data if not present
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                
                self.stemmer = PorterStemmer()
                self.stop_words = set(stopwords.words('english'))
            except Exception:
                pass
        
        # Add technical stop words
        self.stop_words.update({
            'documentation', 'docs', 'guide', 'tutorial', 'example',
            'implementation', 'system', 'component', 'service', 'api',
            'function', 'method', 'class', 'interface', 'module'
        })
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for consistent processing."""
        # Convert to lowercase
        text = text.lower()
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Normalize text first
        text = self.normalize_text(text)
        
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text)
            except Exception:
                # Fallback to simple tokenization
                tokens = re.findall(r'\b\w+\b', text)
        else:
            # Simple tokenization
            tokens = re.findall(r'\b\w+\b', text)
        
        # Filter out stop words and short tokens
        tokens = [
            token for token in tokens 
            if len(token) > 2 and token not in self.stop_words
        ]
        
        return tokens
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Apply stemming to tokens."""
        if self.stemmer:
            return [self.stemmer.stem(token) for token in tokens]
        return tokens
    
    def extract_keywords(self, text: str, max_keywords: int = 20) -> List[str]:
        """Extract important keywords from text using TF-IDF-like scoring."""
        tokens = self.tokenize(text)
        stemmed_tokens = self.stem_tokens(tokens)
        
        # Count token frequencies
        token_counts = Counter(stemmed_tokens)
        
        # Simple TF-IDF approximation
        total_tokens = len(stemmed_tokens)
        keywords = []
        
        for token, count in token_counts.most_common(max_keywords):
            tf = count / total_tokens
            # Simple scoring based on frequency and length
            score = tf * (len(token) / 10.0)  # Favor longer terms
            keywords.append((token, score))
        
        # Sort by score and return top keywords
        keywords.sort(key=lambda x: x[1], reverse=True)
        return [keyword[0] for keyword, _ in keywords[:max_keywords]]
    
    def create_snippet(self, text: str, query_terms: List[str], max_length: int = 200) -> str:
        """Create a snippet highlighting query terms."""
        text = self.normalize_text(text)
        query_terms = [self.normalize_text(term) for term in query_terms]
        
        # Find the best position for the snippet
        best_position = 0
        best_score = 0
        
        # Look for positions with multiple query terms
        for i in range(0, len(text) - max_length, 20):
            snippet = text[i:i + max_length]
            score = sum(1 for term in query_terms if term in snippet)
            if score > best_score:
                best_score = score
                best_position = i
        
        # Extract snippet
        snippet = text[best_position:best_position + max_length]
        
        # Ensure we don't cut words
        if best_position > 0:
            space_pos = snippet.find(' ')
            if space_pos > 0:
                snippet = snippet[space_pos + 1:]
        
        if len(text) > best_position + max_length:
            last_space = snippet.rfind(' ')
            if last_space > 0:
                snippet = snippet[:last_space]
        
        # Add ellipsis if needed
        if best_position > 0:
            snippet = "..." + snippet
        if len(text) > best_position + len(snippet):
            snippet = snippet + "..."
        
        return snippet.strip()
    
    def highlight_terms(self, text: str, query_terms: List[str]) -> str:
        """Highlight query terms in text."""
        highlighted = text
        
        for term in query_terms:
            # Case-insensitive highlighting
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted = pattern.sub(f'<mark>{term}</mark>', highlighted)
        
        return highlighted


class SearchIndex:
    """Advanced search index with SQLite backend for persistence and performance."""
    
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.text_processor = TextProcessor()
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for search index."""
        self.conn = sqlite3.connect(str(self.index_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Create tables
        self.conn.executescript('''
            -- Documents table
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                url TEXT NOT NULL,
                tags TEXT,  -- JSON array
                metadata TEXT,  -- JSON object
                last_modified TIMESTAMP,
                content_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Terms table for inverted index
            CREATE TABLE IF NOT EXISTS terms (
                term TEXT NOT NULL,
                document_id TEXT NOT NULL,
                frequency INTEGER NOT NULL,
                positions TEXT,  -- JSON array of positions
                PRIMARY KEY (term, document_id),
                FOREIGN KEY (document_id) REFERENCES documents(id)
            );
            
            -- Document statistics
            CREATE TABLE IF NOT EXISTS doc_stats (
                document_id TEXT PRIMARY KEY,
                term_count INTEGER,
                unique_terms INTEGER,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            );
            
            -- Global statistics
            CREATE TABLE IF NOT EXISTS global_stats (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            
            -- Create indexes for performance
            CREATE INDEX IF NOT EXISTS idx_terms_term ON terms(term);
            CREATE INDEX IF NOT EXISTS idx_terms_doc ON terms(document_id);
            CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(doc_type);
            CREATE INDEX IF NOT EXISTS idx_documents_modified ON documents(last_modified);
            
            -- Full-text search virtual table
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                title, content, tags, 
                content='documents', 
                content_rowid='rowid'
            );
            
            -- Triggers to keep FTS table in sync
            CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
                INSERT INTO documents_fts(rowid, title, content, tags) 
                VALUES (new.rowid, new.title, new.content, new.tags);
            END;
            
            CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, title, content, tags) 
                VALUES('delete', old.rowid, old.title, old.content, old.tags);
            END;
            
            CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, title, content, tags) 
                VALUES('delete', old.rowid, old.title, old.content, old.tags);
                INSERT INTO documents_fts(rowid, title, content, tags) 
                VALUES (new.rowid, new.title, new.content, new.tags);
            END;
        ''')
        
        self.conn.commit()
    
    def add_document(self, document: SearchDocument) -> bool:
        """Add or update a document in the search index."""
        try:
            # Check if document already exists and is unchanged
            existing = self.conn.execute(
                'SELECT content_hash FROM documents WHERE id = ?',
                (document.id,)
            ).fetchone()
            
            if existing and existing['content_hash'] == document.content_hash:
                return True  # Document unchanged, skip indexing
            
            # Remove existing document if it exists
            if existing:
                self.remove_document(document.id)
            
            # Insert document
            self.conn.execute('''
                INSERT INTO documents 
                (id, title, content, doc_type, file_path, url, tags, metadata, 
                 last_modified, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                document.id,
                document.title,
                document.content,
                document.doc_type.value,
                document.file_path,
                document.url,
                json.dumps(document.tags),
                json.dumps(document.metadata),
                document.last_modified,
                document.content_hash
            ))
            
            # Index terms
            self._index_document_terms(document)
            
            # Update statistics
            self._update_document_stats(document)
            
            self.conn.commit()
            return True
            
        except Exception as e:
            self.conn.rollback()
            print(f"Error adding document {document.id}: {e}")
            return False
    
    def _index_document_terms(self, document: SearchDocument):
        """Index terms for a document with positions and frequencies."""
        # Combine title and content for indexing
        full_text = f"{document.title} {document.content}"
        
        # Tokenize and get positions
        tokens = self.text_processor.tokenize(full_text)
        stemmed_tokens = self.text_processor.stem_tokens(tokens)
        
        # Count frequencies and track positions
        term_data = defaultdict(lambda: {'frequency': 0, 'positions': []})
        
        for position, (original_token, stemmed_token) in enumerate(zip(tokens, stemmed_tokens)):
            term_data[stemmed_token]['frequency'] += 1
            term_data[stemmed_token]['positions'].append(position)
        
        # Insert terms into database
        for term, data in term_data.items():
            self.conn.execute('''
                INSERT INTO terms (term, document_id, frequency, positions)
                VALUES (?, ?, ?, ?)
            ''', (
                term,
                document.id,
                data['frequency'],
                json.dumps(data['positions'])
            ))
    
    def _update_document_stats(self, document: SearchDocument):
        """Update document statistics."""
        full_text = f"{document.title} {document.content}"
        tokens = self.text_processor.tokenize(full_text)
        unique_terms = len(set(self.text_processor.stem_tokens(tokens)))
        
        self.conn.execute('''
            INSERT OR REPLACE INTO doc_stats (document_id, term_count, unique_terms)
            VALUES (?, ?, ?)
        ''', (document.id, len(tokens), unique_terms))
    
    def remove_document(self, document_id: str) -> bool:
        """Remove a document from the search index."""
        try:
            # Remove from all tables
            self.conn.execute('DELETE FROM terms WHERE document_id = ?', (document_id,))
            self.conn.execute('DELETE FROM doc_stats WHERE document_id = ?', (document_id,))
            self.conn.execute('DELETE FROM documents WHERE id = ?', (document_id,))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            self.conn.rollback()
            print(f"Error removing document {document_id}: {e}")
            return False
    
    def search(self, query: SearchQuery) -> SearchResponse:
        """Perform advanced search with ranking and faceting."""
        start_time = datetime.now()
        
        try:
            # Parse and process query
            query_terms = self._parse_query(query.query)
            
            # Build SQL query with filters
            sql_query, params = self._build_search_query(query, query_terms)
            
            # Execute search
            cursor = self.conn.execute(sql_query, params)
            raw_results = cursor.fetchall()
            
            # Calculate relevance scores and create results
            results = []
            for row in raw_results:
                document = self._row_to_document(row)
                score = self._calculate_relevance_score(document, query_terms, query)
                
                result = SearchResult(
                    document=document,
                    score=score,
                    matched_terms=query_terms
                )
                
                # Generate snippet and highlights if requested
                if query.include_content:
                    result.snippet = self.text_processor.create_snippet(
                        document.content, query_terms
                    )
                
                if query.highlight:
                    result.highlights = self._generate_highlights(
                        document, query_terms
                    )
                
                results.append(result)
            
            # Sort by relevance score
            results.sort(key=lambda x: x.score, reverse=True)
            
            # Apply pagination
            total_count = len(results)
            results = results[query.offset:query.offset + query.limit]
            
            # Generate facets
            facets = self._generate_facets(query, query_terms)
            
            # Generate suggestions for empty results
            suggestions = []
            if not results and query.query:
                suggestions = self._generate_suggestions(query.query)
            
            query_time = (datetime.now() - start_time).total_seconds()
            
            return SearchResponse(
                results=results,
                total_count=total_count,
                query_time=query_time,
                facets=facets,
                suggestions=suggestions,
                query=query
            )
            
        except Exception as e:
            print(f"Search error: {e}")
            return SearchResponse(
                results=[],
                total_count=0,
                query_time=0.0,
                query=query
            )
    
    def _parse_query(self, query_string: str) -> List[str]:
        """Parse search query into terms."""
        if not query_string:
            return []
        
        # Handle quoted phrases
        phrases = re.findall(r'"([^"]*)"', query_string)
        query_without_phrases = re.sub(r'"[^"]*"', '', query_string)
        
        # Tokenize remaining query
        terms = self.text_processor.tokenize(query_without_phrases)
        stemmed_terms = self.text_processor.stem_tokens(terms)
        
        # Add phrases as single terms
        for phrase in phrases:
            phrase_tokens = self.text_processor.tokenize(phrase)
            stemmed_phrase = ' '.join(self.text_processor.stem_tokens(phrase_tokens))
            if stemmed_phrase:
                stemmed_terms.append(stemmed_phrase)
        
        return stemmed_terms
    
    def _build_search_query(self, query: SearchQuery, query_terms: List[str]) -> Tuple[str, List]:
        """Build SQL query for search with filters."""
        base_query = '''
            SELECT DISTINCT d.*, 
                   ds.term_count, ds.unique_terms
            FROM documents d
            LEFT JOIN doc_stats ds ON d.id = ds.document_id
        '''
        
        conditions = []
        params = []
        
        # Add text search conditions
        if query_terms:
            # Use FTS for basic text matching
            fts_query = ' OR '.join(f'"{term}"' for term in query_terms)
            base_query += '''
                JOIN documents_fts fts ON d.rowid = fts.rowid
            '''
            conditions.append('documents_fts MATCH ?')
            params.append(fts_query)
        
        # Add document type filter
        if query.doc_types:
            type_placeholders = ','.join('?' * len(query.doc_types))
            conditions.append(f'd.doc_type IN ({type_placeholders})')
            params.extend([dt.value for dt in query.doc_types])
        
        # Add tag filter
        if query.tags:
            for tag in query.tags:
                conditions.append('d.tags LIKE ?')
                params.append(f'%"{tag}"%')
        
        # Add date filters
        if query.date_from:
            conditions.append('d.last_modified >= ?')
            params.append(query.date_from)
        
        if query.date_to:
            conditions.append('d.last_modified <= ?')
            params.append(query.date_to)
        
        # Combine conditions
        if conditions:
            base_query += ' WHERE ' + ' AND '.join(conditions)
        
        # Add ordering and limits
        base_query += ' ORDER BY d.last_modified DESC'
        
        return base_query, params
    
    def _calculate_relevance_score(self, document: SearchDocument, 
                                 query_terms: List[str], query: SearchQuery) -> float:
        """Calculate relevance score using TF-IDF-like algorithm."""
        if not query_terms:
            return 1.0
        
        score = 0.0
        
        # Get document statistics
        doc_stats = self.conn.execute(
            'SELECT term_count, unique_terms FROM doc_stats WHERE document_id = ?',
            (document.id,)
        ).fetchone()
        
        if not doc_stats:
            return 0.0
        
        doc_length = doc_stats['term_count'] or 1
        
        # Calculate score for each query term
        for term in query_terms:
            # Get term frequency in document
            term_data = self.conn.execute(
                'SELECT frequency FROM terms WHERE document_id = ? AND term = ?',
                (document.id, term)
            ).fetchone()
            
            if term_data:
                tf = term_data['frequency'] / doc_length
                
                # Get document frequency for IDF calculation
                df = self.conn.execute(
                    'SELECT COUNT(DISTINCT document_id) FROM terms WHERE term = ?',
                    (term,)
                ).fetchone()[0]
                
                # Total number of documents
                total_docs = self.conn.execute(
                    'SELECT COUNT(*) FROM documents'
                ).fetchone()[0]
                
                if df > 0 and total_docs > 0:
                    idf = math.log(total_docs / df)
                    score += tf * idf
        
        # Boost score for title matches
        title_lower = document.title.lower()
        for term in query_terms:
            if term in title_lower:
                score *= 1.5
        
        # Boost score for exact phrase matches
        content_lower = document.content.lower()
        original_query = query.query.lower()
        if original_query in title_lower:
            score *= 2.0
        elif original_query in content_lower:
            score *= 1.3
        
        return score
    
    def _generate_highlights(self, document: SearchDocument, 
                           query_terms: List[str]) -> List[str]:
        """Generate highlighted snippets for search results."""
        highlights = []
        
        # Highlight in title
        if any(term in document.title.lower() for term in query_terms):
            highlighted_title = self.text_processor.highlight_terms(
                document.title, query_terms
            )
            highlights.append(highlighted_title)
        
        # Find and highlight relevant content sections
        content_sections = document.content.split('\n\n')
        for section in content_sections[:3]:  # Limit to first 3 sections
            section_lower = section.lower()
            if any(term in section_lower for term in query_terms):
                snippet = self.text_processor.create_snippet(section, query_terms, 150)
                highlighted_snippet = self.text_processor.highlight_terms(
                    snippet, query_terms
                )
                highlights.append(highlighted_snippet)
        
        return highlights[:5]  # Limit to 5 highlights
    
    def _generate_facets(self, query: SearchQuery, query_terms: List[str]) -> Dict[str, Dict[str, int]]:
        """Generate faceted search results for filtering."""
        facets = {}
        
        # Document type facets
        type_counts = self.conn.execute('''
            SELECT doc_type, COUNT(*) as count
            FROM documents
            GROUP BY doc_type
            ORDER BY count DESC
        ''').fetchall()
        
        facets['doc_types'] = {
            row['doc_type']: row['count'] for row in type_counts
        }
        
        # Tag facets
        all_tags = []
        tag_rows = self.conn.execute('SELECT tags FROM documents WHERE tags IS NOT NULL').fetchall()
        
        for row in tag_rows:
            try:
                tags = json.loads(row['tags'])
                all_tags.extend(tags)
            except (json.JSONDecodeError, TypeError):
                continue
        
        tag_counts = Counter(all_tags)
        facets['tags'] = dict(tag_counts.most_common(20))
        
        # Date facets (by month)
        date_counts = self.conn.execute('''
            SELECT strftime('%Y-%m', last_modified) as month, COUNT(*) as count
            FROM documents
            WHERE last_modified IS NOT NULL
            GROUP BY month
            ORDER BY month DESC
            LIMIT 12
        ''').fetchall()
        
        facets['dates'] = {
            row['month']: row['count'] for row in date_counts
        }
        
        return facets
    
    def _generate_suggestions(self, query: str) -> List[str]:
        """Generate search suggestions for failed queries."""
        suggestions = []
        
        # Get most common terms from index
        common_terms = self.conn.execute('''
            SELECT term, COUNT(*) as frequency
            FROM terms
            GROUP BY term
            ORDER BY frequency DESC
            LIMIT 100
        ''').fetchall()
        
        query_lower = query.lower()
        
        # Find similar terms using simple string matching
        for row in common_terms:
            term = row['term']
            
            # Exact substring match
            if query_lower in term or term in query_lower:
                suggestions.append(term)
            
            # Simple edit distance approximation
            elif len(query_lower) > 3 and len(term) > 3:
                # Check if terms share significant characters
                common_chars = set(query_lower) & set(term)
                if len(common_chars) >= min(len(query_lower), len(term)) * 0.6:
                    suggestions.append(term)
        
        return suggestions[:5]
    
    def _row_to_document(self, row) -> SearchDocument:
        """Convert database row to SearchDocument object."""
        try:
            tags = json.loads(row['tags']) if row['tags'] else []
        except (json.JSONDecodeError, TypeError):
            tags = []
        
        try:
            metadata = json.loads(row['metadata']) if row['metadata'] else {}
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        
        return SearchDocument(
            id=row['id'],
            title=row['title'],
            content=row['content'],
            doc_type=DocumentType(row['doc_type']),
            file_path=row['file_path'],
            url=row['url'],
            tags=tags,
            metadata=metadata,
            last_modified=datetime.fromisoformat(row['last_modified']) if row['last_modified'] else datetime.now(),
            content_hash=row['content_hash']
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search index statistics."""
        stats = {}
        
        # Document counts by type
        type_counts = self.conn.execute('''
            SELECT doc_type, COUNT(*) as count
            FROM documents
            GROUP BY doc_type
        ''').fetchall()
        
        stats['document_counts'] = {
            row['doc_type']: row['count'] for row in type_counts
        }
        
        # Total documents
        total_docs = self.conn.execute('SELECT COUNT(*) FROM documents').fetchone()[0]
        stats['total_documents'] = total_docs
        
        # Total terms
        total_terms = self.conn.execute('SELECT COUNT(DISTINCT term) FROM terms').fetchone()[0]
        stats['total_terms'] = total_terms
        
        # Index size
        stats['index_size'] = self.index_path.stat().st_size if self.index_path.exists() else 0
        
        # Last update
        last_update = self.conn.execute(
            'SELECT MAX(last_modified) FROM documents'
        ).fetchone()[0]
        stats['last_update'] = last_update
        
        return stats
    
    def close(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()
    
    def __del__(self):
        """Cleanup on destruction."""
        self.close()


class DocumentationSearchEngine:
    """
    Main search engine for documentation system.
    
    Provides high-level interface for indexing and searching documentation
    with comprehensive features including faceted search, relevance ranking,
    and intelligent query processing.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.docs_dir = project_root / "docs"
        self.cache_dir = project_root / ".kiro" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize search index
        self.index = SearchIndex(self.cache_dir / "search_index.db")
        
        # Document type mapping
        self.type_mapping = {
            'api': DocumentType.API,
            'architecture': DocumentType.ARCHITECTURE,
            'algorithms': DocumentType.ALGORITHM,
            'workflows': DocumentType.WORKFLOW,
            'interfaces': DocumentType.INTERFACE,
            'deployment': DocumentType.DEPLOYMENT,
            'frontend': DocumentType.FRONTEND,
            'database': DocumentType.DATABASE
        }
    
    def index_documentation(self, force_reindex: bool = False) -> Dict[str, Any]:
        """Index all documentation files."""
        print("🔍 Indexing documentation for search...")
        
        indexed_count = 0
        skipped_count = 0
        error_count = 0
        
        # Index all markdown files in docs directory
        for md_file in self.docs_dir.rglob("*.md"):
            try:
                # Skip certain files
                if any(skip in str(md_file) for skip in ['.git', '__pycache__', 'node_modules']):
                    continue
                
                document = self._create_document_from_file(md_file)
                if document:
                    if self.index.add_document(document):
                        indexed_count += 1
                    else:
                        skipped_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                print(f"Error indexing {md_file}: {e}")
                error_count += 1
        
        # Index API documentation if available
        api_dir = self.docs_dir / "api"
        if api_dir.exists():
            for api_file in api_dir.rglob("*.json"):
                try:
                    document = self._create_document_from_api_file(api_file)
                    if document:
                        if self.index.add_document(document):
                            indexed_count += 1
                        else:
                            skipped_count += 1
                except Exception as e:
                    print(f"Error indexing API file {api_file}: {e}")
                    error_count += 1
        
        result = {
            'indexed': indexed_count,
            'skipped': skipped_count,
            'errors': error_count,
            'total_processed': indexed_count + skipped_count + error_count
        }
        
        print(f"  ✅ Indexed {indexed_count} documents")
        if skipped_count > 0:
            print(f"  ⏭️  Skipped {skipped_count} unchanged documents")
        if error_count > 0:
            print(f"  ❌ Failed to index {error_count} documents")
        
        return result
    
    def _create_document_from_file(self, file_path: Path) -> Optional[SearchDocument]:
        """Create a SearchDocument from a markdown file."""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Extract title from first heading or filename
            title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            title = title_match.group(1) if title_match else file_path.stem.replace('-', ' ').title()
            
            # Determine document type from path
            doc_type = self._determine_document_type(file_path)
            
            # Extract tags from content
            tags = self._extract_tags_from_content(content, file_path)
            
            # Create relative URL
            relative_path = file_path.relative_to(self.docs_dir)
            url = f"/docs/{relative_path.as_posix()}"
            
            # Extract metadata
            metadata = {
                'file_size': file_path.stat().st_size,
                'word_count': len(content.split()),
                'line_count': len(content.splitlines())
            }
            
            # Get last modified time
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
    
    def _create_document_from_api_file(self, file_path: Path) -> Optional[SearchDocument]:
        """Create a SearchDocument from an API specification file."""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            if file_path.suffix == '.json':
                # Parse OpenAPI/Swagger JSON
                api_spec = json.loads(content)
                title = api_spec.get('info', {}).get('title', 'API Documentation')
                
                # Extract searchable content from API spec
                searchable_content = self._extract_api_content(api_spec)
                
                tags = ['api', 'openapi', 'swagger']
                if 'tags' in api_spec:
                    tags.extend([tag.get('name', '') for tag in api_spec['tags']])
                
                metadata = {
                    'api_version': api_spec.get('info', {}).get('version', ''),
                    'endpoints_count': len(api_spec.get('paths', {})),
                    'file_size': file_path.stat().st_size
                }
                
            else:
                # Fallback for other file types
                title = file_path.stem.replace('-', ' ').title()
                searchable_content = content
                tags = ['api']
                metadata = {'file_size': file_path.stat().st_size}
            
            relative_path = file_path.relative_to(self.docs_dir)
            url = f"/docs/{relative_path.as_posix()}"
            last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
            
            return SearchDocument(
                id=str(file_path.relative_to(self.project_root)),
                title=title,
                content=searchable_content,
                doc_type=DocumentType.API,
                file_path=str(file_path.relative_to(self.project_root)),
                url=url,
                tags=tags,
                metadata=metadata,
                last_modified=last_modified
            )
            
        except Exception as e:
            print(f"Error creating API document from {file_path}: {e}")
            return None
    
    def _determine_document_type(self, file_path: Path) -> DocumentType:
        """Determine document type from file path."""
        path_str = str(file_path).lower()
        
        for key, doc_type in self.type_mapping.items():
            if key in path_str:
                return doc_type
        
        return DocumentType.GENERAL
    
    def _extract_tags_from_content(self, content: str, file_path: Path) -> List[str]:
        """Extract tags from document content and path."""
        tags = []
        
        # Add tags based on file path
        path_parts = file_path.parts
        for part in path_parts:
            if part.lower() in self.type_mapping:
                tags.append(part.lower())
        
        # Extract tags from content (look for common patterns)
        # Tags from headings
        headings = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        for heading in headings[:5]:  # Limit to first 5 headings
            # Extract meaningful words from headings
            words = re.findall(r'\b\w+\b', heading.lower())
            tags.extend([word for word in words if len(word) > 3])
        
        # Remove duplicates and limit
        tags = list(set(tags))[:10]
        
        return tags
    
    def _extract_api_content(self, api_spec: Dict[str, Any]) -> str:
        """Extract searchable content from API specification."""
        content_parts = []
        
        # Add info section
        info = api_spec.get('info', {})
        if 'title' in info:
            content_parts.append(info['title'])
        if 'description' in info:
            content_parts.append(info['description'])
        
        # Add paths and operations
        paths = api_spec.get('paths', {})
        for path, methods in paths.items():
            content_parts.append(f"Endpoint: {path}")
            
            for method, operation in methods.items():
                if isinstance(operation, dict):
                    if 'summary' in operation:
                        content_parts.append(operation['summary'])
                    if 'description' in operation:
                        content_parts.append(operation['description'])
                    if 'tags' in operation:
                        content_parts.extend(operation['tags'])
        
        # Add component schemas
        components = api_spec.get('components', {})
        schemas = components.get('schemas', {})
        for schema_name, schema_def in schemas.items():
            content_parts.append(f"Schema: {schema_name}")
            if isinstance(schema_def, dict) and 'description' in schema_def:
                content_parts.append(schema_def['description'])
        
        return '\n'.join(content_parts)
    
    def search(self, query_string: str, **kwargs) -> SearchResponse:
        """
        Perform search with advanced options.
        
        Args:
            query_string: The search query
            **kwargs: Additional search options (doc_types, tags, limit, etc.)
        
        Returns:
            SearchResponse with results and metadata
        """
        # Create search query object
        query = SearchQuery(
            query=query_string,
            doc_types=[DocumentType(dt) for dt in kwargs.get('doc_types', [])],
            tags=kwargs.get('tags', []),
            limit=kwargs.get('limit', 50),
            offset=kwargs.get('offset', 0),
            include_content=kwargs.get('include_content', True),
            highlight=kwargs.get('highlight', True)
        )
        
        # Handle date filters
        if 'date_from' in kwargs:
            query.date_from = kwargs['date_from']
        if 'date_to' in kwargs:
            query.date_to = kwargs['date_to']
        
        return self.index.search(query)
    
    def get_suggestions(self, partial_query: str, limit: int = 10) -> List[str]:
        """Get search suggestions for autocomplete."""
        if len(partial_query) < 2:
            return []
        
        # Get common terms that start with the partial query
        suggestions = self.index.conn.execute('''
            SELECT DISTINCT term, COUNT(*) as frequency
            FROM terms
            WHERE term LIKE ? || '%'
            GROUP BY term
            ORDER BY frequency DESC
            LIMIT ?
        ''', (partial_query.lower(), limit)).fetchall()
        
        return [row['term'] for row in suggestions]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive search statistics."""
        return self.index.get_statistics()
    
    def close(self):
        """Close search engine and cleanup resources."""
        self.index.close()


# CLI interface for testing and maintenance
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Documentation Search Engine")
    parser.add_argument("--index", action="store_true", help="Index documentation")
    parser.add_argument("--search", type=str, help="Search query")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--project-root", type=str, default=".", help="Project root directory")
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    search_engine = DocumentationSearchEngine(project_root)
    
    try:
        if args.index:
            result = search_engine.index_documentation()
            print(f"Indexing complete: {result}")
        
        if args.search:
            response = search_engine.search(args.search)
            print(f"Found {response.total_count} results in {response.query_time:.3f}s")
            
            for i, result in enumerate(response.results[:5], 1):
                print(f"\n{i}. {result.document.title}")
                print(f"   Type: {result.document.doc_type.value}")
                print(f"   Score: {result.score:.3f}")
                print(f"   URL: {result.document.url}")
                if result.snippet:
                    print(f"   Snippet: {result.snippet[:100]}...")
        
        if args.stats:
            stats = search_engine.get_statistics()
            print("Search Index Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
    
    finally:
        search_engine.close()