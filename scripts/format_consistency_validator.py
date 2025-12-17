#!/usr/bin/env python3
"""
Format Consistency Validator for Multi-Format Export Engine

This module implements validation to ensure formatting consistency across all export formats,
diagram and mathematical notation preservation, and format-specific optimization while
maintaining content integrity.
"""

import os
import re
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.export_engine import ExportFormat, ExportResult, DocumentSet


@dataclass
class ConsistencyIssue:
    """Represents a consistency issue between formats."""
    issue_type: str
    severity: str  # "error", "warning", "info"
    description: str
    affected_formats: List[ExportFormat]
    recommendation: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FormatAnalysis:
    """Analysis results for a specific export format."""
    format: ExportFormat
    file_path: Path
    file_size: int
    content_elements: Dict[str, int] = field(default_factory=dict)
    structure_elements: Dict[str, bool] = field(default_factory=dict)
    preserved_features: Dict[str, bool] = field(default_factory=dict)
    format_specific_features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsistencyReport:
    """Comprehensive consistency validation report."""
    overall_consistent: bool
    format_analyses: Dict[ExportFormat, FormatAnalysis] = field(default_factory=dict)
    consistency_issues: List[ConsistencyIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    validation_timestamp: datetime = field(default_factory=datetime.now)


class FormatConsistencyValidator:
    """
    Validates formatting consistency across all export formats.
    
    Implements formatting consistency validation across all export formats,
    adds diagram and mathematical notation preservation checking,
    and creates format-specific optimization while maintaining content integrity.
    """
    
    def __init__(self):
        self.supported_formats = {ExportFormat.HTML, ExportFormat.PDF, ExportFormat.EPUB}
        self.content_extractors = {
            ExportFormat.HTML: self._extract_html_content,
            ExportFormat.PDF: self._extract_pdf_content,
            ExportFormat.EPUB: self._extract_epub_content
        }
    
    def validate_consistency(self, export_results: Dict[ExportFormat, ExportResult], 
                           original_document_set: DocumentSet) -> ConsistencyReport:
        """
        Validate consistency across all export formats.
        
        Args:
            export_results: Dictionary of export results by format
            original_document_set: Original document set used for export
            
        Returns:
            ConsistencyReport with detailed analysis
        """
        report = ConsistencyReport(overall_consistent=True)
        
        # Analyze each format
        for format_type, result in export_results.items():
            if result.success and format_type in self.supported_formats:
                analysis = self._analyze_format(format_type, result, original_document_set)
                report.format_analyses[format_type] = analysis
        
        # Perform cross-format consistency checks
        if len(report.format_analyses) > 1:
            consistency_issues = self._check_cross_format_consistency(
                report.format_analyses, original_document_set
            )
            report.consistency_issues.extend(consistency_issues)
        
        # Check content preservation
        content_issues = self._check_content_preservation(
            report.format_analyses, original_document_set
        )
        report.consistency_issues.extend(content_issues)
        
        # Check diagram preservation
        diagram_issues = self._check_diagram_preservation(
            report.format_analyses, original_document_set
        )
        report.consistency_issues.extend(diagram_issues)
        
        # Check mathematical notation preservation
        math_issues = self._check_math_preservation(
            report.format_analyses, original_document_set
        )
        report.consistency_issues.extend(math_issues)
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report.consistency_issues)
        
        # Determine overall consistency
        error_issues = [issue for issue in report.consistency_issues if issue.severity == "error"]
        report.overall_consistent = len(error_issues) == 0
        
        return report
    
    def _analyze_format(self, format_type: ExportFormat, result: ExportResult, 
                       document_set: DocumentSet) -> FormatAnalysis:
        """Analyze a specific export format."""
        analysis = FormatAnalysis(
            format=format_type,
            file_path=result.output_path,
            file_size=result.file_size
        )
        
        # Extract content using format-specific extractor
        if format_type in self.content_extractors:
            try:
                content_data = self.content_extractors[format_type](result.output_path)
                analysis.content_elements = content_data.get('content_elements', {})
                analysis.structure_elements = content_data.get('structure_elements', {})
                analysis.preserved_features = content_data.get('preserved_features', {})
                analysis.format_specific_features = content_data.get('format_specific_features', {})
            except Exception as e:
                print(f"Warning: Failed to analyze {format_type.value}: {e}")
        
        return analysis
    
    def _extract_html_content(self, output_path: Path) -> Dict[str, Any]:
        """Extract content analysis from HTML export."""
        content_data = {
            'content_elements': {},
            'structure_elements': {},
            'preserved_features': {},
            'format_specific_features': {}
        }
        
        if output_path.is_dir():
            # Find main HTML files
            html_files = list(output_path.rglob("*.html"))
            
            total_content = ""
            for html_file in html_files:
                try:
                    total_content += html_file.read_text(encoding='utf-8')
                except Exception:
                    continue
            
            # Analyze content elements
            content_data['content_elements'] = {
                'headings': len(re.findall(r'<h[1-6][^>]*>', total_content, re.IGNORECASE)),
                'paragraphs': len(re.findall(r'<p[^>]*>', total_content, re.IGNORECASE)),
                'lists': len(re.findall(r'<[uo]l[^>]*>', total_content, re.IGNORECASE)),
                'code_blocks': len(re.findall(r'<pre[^>]*>', total_content, re.IGNORECASE)),
                'links': len(re.findall(r'<a[^>]*href', total_content, re.IGNORECASE)),
                'images': len(re.findall(r'<img[^>]*>', total_content, re.IGNORECASE))
            }
            
            # Analyze structure elements
            content_data['structure_elements'] = {
                'has_navigation': bool(re.search(r'<nav[^>]*>', total_content, re.IGNORECASE)),
                'has_search': bool(re.search(r'search', total_content, re.IGNORECASE)),
                'has_toc': bool(re.search(r'table.of.contents|toc', total_content, re.IGNORECASE)),
                'responsive_design': bool(re.search(r'viewport|@media', total_content, re.IGNORECASE)),
                'proper_html_structure': all([
                    '<!DOCTYPE html>' in total_content,
                    '<html' in total_content,
                    '<head>' in total_content,
                    '<body>' in total_content
                ])
            }
            
            # Analyze preserved features
            content_data['preserved_features'] = {
                'diagrams': bool(re.search(r'mermaid|diagram', total_content, re.IGNORECASE)),
                'math': bool(re.search(r'mathjax|\$\$|\\\(', total_content, re.IGNORECASE)),
                'code_highlighting': bool(re.search(r'highlight|prism|code', total_content, re.IGNORECASE)),
                'cross_references': bool(re.search(r'href="#|id="', total_content, re.IGNORECASE))
            }
            
            # HTML-specific features
            content_data['format_specific_features'] = {
                'interactive_elements': len(re.findall(r'<(button|input|select)', total_content, re.IGNORECASE)),
                'css_files': len(list(output_path.rglob("*.css"))),
                'js_files': len(list(output_path.rglob("*.js"))),
                'asset_files': len(list(output_path.rglob("assets/*"))),
                'search_functionality': bool(re.search(r'search.*js|search.*function', total_content, re.IGNORECASE))
            }
        
        return content_data
    
    def _extract_pdf_content(self, output_path: Path) -> Dict[str, Any]:
        """Extract content analysis from PDF export."""
        content_data = {
            'content_elements': {},
            'structure_elements': {},
            'preserved_features': {},
            'format_specific_features': {}
        }
        
        if output_path.is_dir():
            pdf_files = list(output_path.glob("*.pdf"))
        else:
            pdf_files = [output_path] if output_path.suffix == '.pdf' else []
        
        if pdf_files:
            pdf_file = pdf_files[0]
            file_size = pdf_file.stat().st_size
            
            try:
                # Basic PDF analysis (without external libraries)
                pdf_content = pdf_file.read_bytes()
                pdf_text = pdf_content.decode('utf-8', errors='ignore')
                
                # Analyze content elements (estimated from PDF structure)
                content_data['content_elements'] = {
                    'estimated_pages': pdf_text.count('/Type /Page'),
                    'text_content': len(re.findall(r'[a-zA-Z]+', pdf_text)),
                    'structure_markers': pdf_text.count('/Type'),
                }
                
                # Analyze structure elements
                content_data['structure_elements'] = {
                    'has_toc': bool(re.search(r'table.of.contents|toc|outline', pdf_text, re.IGNORECASE)),
                    'proper_pdf_structure': pdf_text.startswith('%PDF-'),
                    'has_metadata': bool(re.search(r'/Title|/Author|/Subject', pdf_text)),
                    'bookmarks': bool(re.search(r'/Outline|/Dest', pdf_text))
                }
                
                # Analyze preserved features
                content_data['preserved_features'] = {
                    'diagrams': bool(re.search(r'diagram|figure|image', pdf_text, re.IGNORECASE)),
                    'math': bool(re.search(r'equation|formula|math', pdf_text, re.IGNORECASE)),
                    'formatting': file_size > 5000,  # Substantial content suggests formatting
                    'cross_references': bool(re.search(r'/Link|/Annot', pdf_text))
                }
                
                # PDF-specific features
                content_data['format_specific_features'] = {
                    'file_size': file_size,
                    'pdf_version': re.search(r'%PDF-(\d+\.\d+)', pdf_text),
                    'compression': bool(re.search(r'/Filter', pdf_text)),
                    'fonts': len(re.findall(r'/Font', pdf_text)),
                    'print_optimized': True  # PDFs are inherently print-optimized
                }
                
            except Exception as e:
                print(f"Warning: PDF analysis failed: {e}")
                # Fallback analysis
                content_data['structure_elements']['proper_pdf_structure'] = pdf_file.exists()
                content_data['format_specific_features']['file_size'] = file_size
        
        return content_data
    
    def _extract_epub_content(self, output_path: Path) -> Dict[str, Any]:
        """Extract content analysis from EPUB export."""
        content_data = {
            'content_elements': {},
            'structure_elements': {},
            'preserved_features': {},
            'format_specific_features': {}
        }
        
        if output_path.is_dir():
            epub_files = list(output_path.glob("*.epub"))
        else:
            epub_files = [output_path] if output_path.suffix == '.epub' else []
        
        if epub_files:
            epub_file = epub_files[0]
            
            try:
                with zipfile.ZipFile(epub_file, 'r') as epub_zip:
                    file_list = epub_zip.namelist()
                    
                    # Read content files
                    total_content = ""
                    for file_name in file_list:
                        if file_name.endswith(('.html', '.xhtml')):
                            try:
                                content = epub_zip.read(file_name).decode('utf-8')
                                total_content += content
                            except Exception:
                                continue
                    
                    # Analyze content elements
                    content_data['content_elements'] = {
                        'chapters': len([f for f in file_list if 'chapter' in f.lower()]),
                        'headings': len(re.findall(r'<h[1-6][^>]*>', total_content, re.IGNORECASE)),
                        'paragraphs': len(re.findall(r'<p[^>]*>', total_content, re.IGNORECASE)),
                        'lists': len(re.findall(r'<[uo]l[^>]*>', total_content, re.IGNORECASE)),
                        'content_files': len([f for f in file_list if f.endswith(('.html', '.xhtml'))])
                    }
                    
                    # Analyze structure elements
                    content_data['structure_elements'] = {
                        'has_mimetype': 'mimetype' in file_list,
                        'has_container': any('container.xml' in f for f in file_list),
                        'has_opf': any(f.endswith('.opf') for f in file_list),
                        'has_navigation': any('nav' in f.lower() for f in file_list),
                        'has_toc': any('toc' in f.lower() for f in file_list),
                        'proper_epub_structure': all([
                            'mimetype' in file_list,
                            any('META-INF' in f for f in file_list),
                            any('OEBPS' in f or 'OPS' in f for f in file_list)
                        ])
                    }
                    
                    # Analyze preserved features
                    content_data['preserved_features'] = {
                        'diagrams': bool(re.search(r'diagram|figure|svg', total_content, re.IGNORECASE)),
                        'math': bool(re.search(r'math|equation|formula', total_content, re.IGNORECASE)),
                        'formatting': bool(re.search(r'<style|\.css', total_content, re.IGNORECASE)),
                        'cross_references': bool(re.search(r'href=|id=', total_content, re.IGNORECASE))
                    }
                    
                    # EPUB-specific features
                    content_data['format_specific_features'] = {
                        'file_size': epub_file.stat().st_size,
                        'total_files': len(file_list),
                        'css_files': len([f for f in file_list if f.endswith('.css')]),
                        'image_files': len([f for f in file_list if any(f.endswith(ext) for ext in ['.jpg', '.png', '.gif', '.svg'])]),
                        'epub_version': 3 if any('epub:type' in total_content for _ in [1]) else 2,
                        'reader_compatible': content_data['structure_elements']['proper_epub_structure']
                    }
                    
            except Exception as e:
                print(f"Warning: EPUB analysis failed: {e}")
                # Fallback analysis
                content_data['structure_elements']['proper_epub_structure'] = epub_file.exists()
                content_data['format_specific_features']['file_size'] = epub_file.stat().st_size
        
        return content_data
    
    def _check_cross_format_consistency(self, format_analyses: Dict[ExportFormat, FormatAnalysis], 
                                       document_set: DocumentSet) -> List[ConsistencyIssue]:
        """Check consistency across different export formats."""
        issues = []
        
        if len(format_analyses) < 2:
            return issues
        
        # Check content element consistency
        content_consistency = self._check_content_element_consistency(format_analyses)
        issues.extend(content_consistency)
        
        # Check structure consistency
        structure_consistency = self._check_structure_consistency(format_analyses)
        issues.extend(structure_consistency)
        
        # Check file size consistency (relative to format expectations)
        size_consistency = self._check_file_size_consistency(format_analyses)
        issues.extend(size_consistency)
        
        return issues
    
    def _check_content_element_consistency(self, format_analyses: Dict[ExportFormat, FormatAnalysis]) -> List[ConsistencyIssue]:
        """Check consistency of content elements across formats."""
        issues = []
        
        # Get content element counts for comparison
        element_counts = {}
        for format_type, analysis in format_analyses.items():
            element_counts[format_type] = analysis.content_elements
        
        # Check heading consistency
        heading_counts = {fmt: counts.get('headings', 0) for fmt, counts in element_counts.items()}
        if len(set(heading_counts.values())) > 1:
            # Allow some variation due to format differences
            max_count = max(heading_counts.values())
            min_count = min(heading_counts.values())
            
            if max_count > 0 and (max_count - min_count) / max_count > 0.3:  # More than 30% difference
                # Convert ExportFormat to string for JSON serialization
                heading_counts_str = {fmt.value: count for fmt, count in heading_counts.items()}
                issues.append(ConsistencyIssue(
                    issue_type="heading_count_inconsistency",
                    severity="warning",
                    description=f"Heading counts vary significantly across formats: {heading_counts_str}",
                    affected_formats=list(format_analyses.keys()),
                    recommendation="Review heading structure in format-specific templates",
                    details={'heading_counts': heading_counts_str}
                ))
        
        # Check paragraph consistency
        paragraph_counts = {fmt: counts.get('paragraphs', 0) for fmt, counts in element_counts.items()}
        if len(set(paragraph_counts.values())) > 1:
            max_count = max(paragraph_counts.values())
            min_count = min(paragraph_counts.values())
            
            if max_count > 0 and (max_count - min_count) / max_count > 0.2:  # More than 20% difference
                # Convert ExportFormat to string for JSON serialization
                paragraph_counts_str = {fmt.value: count for fmt, count in paragraph_counts.items()}
                issues.append(ConsistencyIssue(
                    issue_type="paragraph_count_inconsistency",
                    severity="info",
                    description=f"Paragraph counts vary across formats: {paragraph_counts_str}",
                    affected_formats=list(format_analyses.keys()),
                    recommendation="This may be normal due to format-specific rendering differences",
                    details={'paragraph_counts': paragraph_counts_str}
                ))
        
        return issues
    
    def _check_structure_consistency(self, format_analyses: Dict[ExportFormat, FormatAnalysis]) -> List[ConsistencyIssue]:
        """Check structural consistency across formats."""
        issues = []
        
        # Check navigation consistency
        nav_support = {}
        for format_type, analysis in format_analyses.items():
            if format_type == ExportFormat.HTML:
                nav_support[format_type] = analysis.structure_elements.get('has_navigation', False)
            elif format_type == ExportFormat.PDF:
                nav_support[format_type] = analysis.structure_elements.get('has_toc', False)
            elif format_type == ExportFormat.EPUB:
                nav_support[format_type] = analysis.structure_elements.get('has_navigation', False)
        
        if nav_support and not all(nav_support.values()):
            missing_nav = [fmt for fmt, has_nav in nav_support.items() if not has_nav]
            issues.append(ConsistencyIssue(
                issue_type="navigation_inconsistency",
                severity="warning",
                description=f"Navigation/TOC missing in some formats: {[fmt.value for fmt in missing_nav]}",
                affected_formats=missing_nav,
                recommendation="Ensure all formats include appropriate navigation structures",
                details={'navigation_support': {fmt.value: has_nav for fmt, has_nav in nav_support.items()}}
            ))
        
        # Check proper format structure
        structure_issues = []
        for format_type, analysis in format_analyses.items():
            if format_type == ExportFormat.HTML:
                if not analysis.structure_elements.get('proper_html_structure', False):
                    structure_issues.append(format_type)
            elif format_type == ExportFormat.PDF:
                if not analysis.structure_elements.get('proper_pdf_structure', False):
                    structure_issues.append(format_type)
            elif format_type == ExportFormat.EPUB:
                if not analysis.structure_elements.get('proper_epub_structure', False):
                    structure_issues.append(format_type)
        
        if structure_issues:
            issues.append(ConsistencyIssue(
                issue_type="format_structure_invalid",
                severity="error",
                description=f"Invalid format structure detected in: {[fmt.value for fmt in structure_issues]}",
                affected_formats=structure_issues,
                recommendation="Fix format-specific structure issues to ensure proper rendering",
                details={'invalid_formats': [fmt.value for fmt in structure_issues]}
            ))
        
        return issues
    
    def _check_file_size_consistency(self, format_analyses: Dict[ExportFormat, FormatAnalysis]) -> List[ConsistencyIssue]:
        """Check file size consistency relative to format expectations."""
        issues = []
        
        file_sizes = {fmt: analysis.file_size for fmt, analysis in format_analyses.items()}
        
        # Expected size relationships (HTML > EPUB > PDF for similar content)
        # But allow for significant variation due to format differences
        
        # Check for unusually small files
        for format_type, size in file_sizes.items():
            if size < 1000:  # Less than 1KB
                issues.append(ConsistencyIssue(
                    issue_type="file_size_too_small",
                    severity="warning",
                    description=f"{format_type.value} export is unusually small ({size} bytes)",
                    affected_formats=[format_type],
                    recommendation="Check if content was properly exported to this format",
                    details={'file_size': size}
                ))
        
        # Check for extreme size differences
        if len(file_sizes) > 1:
            max_size = max(file_sizes.values())
            min_size = min(file_sizes.values())
            
            if max_size > 0 and min_size > 0:
                size_ratio = max_size / min_size
                if size_ratio > 50:  # One format is 50x larger than another
                    issues.append(ConsistencyIssue(
                        issue_type="extreme_size_difference",
                        severity="info",
                        description=f"Large size difference between formats: {file_sizes}",
                        affected_formats=list(format_analyses.keys()),
                        recommendation="Review if size differences are expected for the content type",
                        details={'file_sizes': {fmt.value: size for fmt, size in file_sizes.items()}}
                    ))
        
        return issues
    
    def _check_content_preservation(self, format_analyses: Dict[ExportFormat, FormatAnalysis], 
                                  document_set: DocumentSet) -> List[ConsistencyIssue]:
        """Check that content is preserved across formats."""
        issues = []
        
        # Check if all formats preserve basic content
        content_preservation = {}
        for format_type, analysis in format_analyses.items():
            preserved = analysis.preserved_features
            content_preservation[format_type] = {
                'has_text_content': (
                    analysis.content_elements.get('paragraphs', 0) > 0 or
                    analysis.content_elements.get('headings', 0) > 0
                ),
                'preserves_structure': (
                    analysis.content_elements.get('headings', 0) > 0 and
                    analysis.content_elements.get('paragraphs', 0) > 0
                )
            }
        
        # Check for formats missing content
        missing_content = []
        for format_type, preservation in content_preservation.items():
            if not preservation['has_text_content']:
                missing_content.append(format_type)
        
        if missing_content:
            issues.append(ConsistencyIssue(
                issue_type="content_not_preserved",
                severity="error",
                description=f"Content appears to be missing in formats: {[fmt.value for fmt in missing_content]}",
                affected_formats=missing_content,
                recommendation="Check content extraction and conversion processes for these formats",
                details={'missing_content_formats': [fmt.value for fmt in missing_content]}
            ))
        
        # Check for structure preservation
        missing_structure = []
        for format_type, preservation in content_preservation.items():
            if not preservation['preserves_structure']:
                missing_structure.append(format_type)
        
        if missing_structure:
            issues.append(ConsistencyIssue(
                issue_type="structure_not_preserved",
                severity="warning",
                description=f"Document structure may not be preserved in: {[fmt.value for fmt in missing_structure]}",
                affected_formats=missing_structure,
                recommendation="Review heading and paragraph structure in these formats",
                details={'missing_structure_formats': [fmt.value for fmt in missing_structure]}
            ))
        
        return issues
    
    def _check_diagram_preservation(self, format_analyses: Dict[ExportFormat, FormatAnalysis], 
                                  document_set: DocumentSet) -> List[ConsistencyIssue]:
        """Check diagram preservation across formats."""
        issues = []
        
        # Check if original documents contain diagrams
        has_diagrams = any(
            len(doc.diagrams) > 0 for doc in document_set.documents
        )
        
        if not has_diagrams:
            return issues  # No diagrams to check
        
        # Check diagram preservation in each format
        diagram_preservation = {}
        for format_type, analysis in format_analyses.items():
            diagram_preservation[format_type] = analysis.preserved_features.get('diagrams', False)
        
        # Check for formats not preserving diagrams
        missing_diagrams = [fmt for fmt, preserved in diagram_preservation.items() if not preserved]
        
        if missing_diagrams:
            # This is expected for some formats (like basic PDF), so it's a warning not an error
            issues.append(ConsistencyIssue(
                issue_type="diagrams_not_preserved",
                severity="warning",
                description=f"Diagrams may not be preserved in formats: {[fmt.value for fmt in missing_diagrams]}",
                affected_formats=missing_diagrams,
                recommendation="Consider adding diagram placeholders or alternative representations for these formats",
                details={
                    'missing_diagram_formats': [fmt.value for fmt in missing_diagrams],
                    'diagram_preservation': {fmt.value: preserved for fmt, preserved in diagram_preservation.items()}
                }
            ))
        
        return issues
    
    def _check_math_preservation(self, format_analyses: Dict[ExportFormat, FormatAnalysis], 
                               document_set: DocumentSet) -> List[ConsistencyIssue]:
        """Check mathematical notation preservation across formats."""
        issues = []
        
        # Check if original documents contain math
        has_math = any(
            len(doc.math_expressions) > 0 for doc in document_set.documents
        )
        
        if not has_math:
            return issues  # No math to check
        
        # Check math preservation in each format
        math_preservation = {}
        for format_type, analysis in format_analyses.items():
            math_preservation[format_type] = analysis.preserved_features.get('math', False)
        
        # Check for formats not preserving math
        missing_math = [fmt for fmt, preserved in math_preservation.items() if not preserved]
        
        if missing_math:
            issues.append(ConsistencyIssue(
                issue_type="math_not_preserved",
                severity="warning",
                description=f"Mathematical notation may not be preserved in formats: {[fmt.value for fmt in missing_math]}",
                affected_formats=missing_math,
                recommendation="Consider adding LaTeX rendering or alternative math representations for these formats",
                details={
                    'missing_math_formats': [fmt.value for fmt in missing_math],
                    'math_preservation': {fmt.value: preserved for fmt, preserved in math_preservation.items()}
                }
            ))
        
        return issues
    
    def _generate_recommendations(self, consistency_issues: List[ConsistencyIssue]) -> List[str]:
        """Generate overall recommendations based on consistency issues."""
        recommendations = []
        
        # Count issues by type
        issue_types = {}
        for issue in consistency_issues:
            issue_types[issue.issue_type] = issue_types.get(issue.issue_type, 0) + 1
        
        # Generate recommendations based on common issues
        if issue_types.get('content_not_preserved', 0) > 0:
            recommendations.append("Review content extraction processes to ensure all formats receive complete content")
        
        if issue_types.get('structure_not_preserved', 0) > 0:
            recommendations.append("Verify that document structure (headings, paragraphs) is properly converted in all formats")
        
        if issue_types.get('navigation_inconsistency', 0) > 0:
            recommendations.append("Implement consistent navigation/table of contents across all export formats")
        
        if issue_types.get('diagrams_not_preserved', 0) > 0:
            recommendations.append("Consider adding diagram rendering or placeholders for formats that don't support interactive diagrams")
        
        if issue_types.get('math_not_preserved', 0) > 0:
            recommendations.append("Implement mathematical notation rendering or provide alternative representations for all formats")
        
        if issue_types.get('format_structure_invalid', 0) > 0:
            recommendations.append("Fix format-specific structural issues to ensure proper rendering in target applications")
        
        # General recommendations
        error_count = len([issue for issue in consistency_issues if issue.severity == "error"])
        warning_count = len([issue for issue in consistency_issues if issue.severity == "warning"])
        
        if error_count > 0:
            recommendations.append(f"Address {error_count} critical consistency errors to ensure proper format compatibility")
        
        if warning_count > 3:
            recommendations.append(f"Consider addressing {warning_count} consistency warnings to improve format uniformity")
        
        if not recommendations:
            recommendations.append("Export formats show good consistency - continue monitoring for future exports")
        
        return recommendations
    
    def generate_consistency_report(self, consistency_report: ConsistencyReport, 
                                  output_path: Optional[Path] = None) -> str:
        """Generate a detailed consistency report."""
        report_content = f"""# Multi-Format Export Consistency Report

**Generated**: {consistency_report.validation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}
**Overall Status**: {'✅ CONSISTENT' if consistency_report.overall_consistent else '❌ INCONSISTENT'}

## Summary

- **Formats Analyzed**: {len(consistency_report.format_analyses)}
- **Issues Found**: {len(consistency_report.consistency_issues)}
- **Errors**: {len([i for i in consistency_report.consistency_issues if i.severity == 'error'])}
- **Warnings**: {len([i for i in consistency_report.consistency_issues if i.severity == 'warning'])}

## Format Analysis

"""
        
        for format_type, analysis in consistency_report.format_analyses.items():
            report_content += f"""### {format_type.value.upper()} Format

**File**: `{analysis.file_path}`
**Size**: {analysis.file_size:,} bytes

**Content Elements**:
"""
            for element, count in analysis.content_elements.items():
                report_content += f"- {element.replace('_', ' ').title()}: {count}\n"
            
            report_content += "\n**Structure Elements**:\n"
            for element, present in analysis.structure_elements.items():
                status = "✅" if present else "❌"
                report_content += f"- {element.replace('_', ' ').title()}: {status}\n"
            
            report_content += "\n**Preserved Features**:\n"
            for feature, preserved in analysis.preserved_features.items():
                status = "✅" if preserved else "❌"
                report_content += f"- {feature.replace('_', ' ').title()}: {status}\n"
            
            report_content += "\n"
        
        # Add issues section
        if consistency_report.consistency_issues:
            report_content += "## Consistency Issues\n\n"
            
            for i, issue in enumerate(consistency_report.consistency_issues, 1):
                severity_icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}.get(issue.severity, "⚪")
                report_content += f"### Issue {i}: {issue.issue_type.replace('_', ' ').title()} {severity_icon}\n\n"
                report_content += f"**Severity**: {issue.severity.upper()}\n"
                report_content += f"**Description**: {issue.description}\n"
                report_content += f"**Affected Formats**: {', '.join(fmt.value for fmt in issue.affected_formats)}\n"
                
                if issue.recommendation:
                    report_content += f"**Recommendation**: {issue.recommendation}\n"
                
                if issue.details:
                    report_content += f"**Details**: {json.dumps(issue.details, indent=2)}\n"
                
                report_content += "\n"
        
        # Add recommendations section
        if consistency_report.recommendations:
            report_content += "## Recommendations\n\n"
            for i, recommendation in enumerate(consistency_report.recommendations, 1):
                report_content += f"{i}. {recommendation}\n"
            report_content += "\n"
        
        # Add next steps
        report_content += """## Next Steps

1. **Address Critical Issues**: Fix all errors marked as 'error' severity first
2. **Review Warnings**: Consider addressing warnings to improve consistency
3. **Implement Recommendations**: Follow the recommendations to enhance format uniformity
4. **Re-validate**: After making changes, re-run consistency validation
5. **Monitor**: Set up regular consistency checks for future exports

## Validation Criteria

### Content Preservation
- All formats should contain the same essential content
- Heading and paragraph counts should be reasonably consistent
- Text content should be preserved across all formats

### Structure Consistency
- Each format should have proper format-specific structure
- Navigation/TOC should be available in appropriate forms
- Cross-references should be maintained where possible

### Feature Preservation
- Diagrams should be rendered or have appropriate placeholders
- Mathematical notation should be preserved or have alternatives
- Code blocks and formatting should be maintained

### Format Optimization
- Each format should be optimized for its intended use case
- File sizes should be reasonable for the content amount
- Format-specific features should be properly implemented
"""
        
        # Save report if output path provided
        if output_path:
            output_path.write_text(report_content, encoding='utf-8')
        
        return report_content


def validate_export_consistency(export_results: Dict[ExportFormat, ExportResult], 
                               document_set: DocumentSet,
                               output_report_path: Optional[Path] = None) -> ConsistencyReport:
    """
    Convenience function to validate export consistency and generate report.
    
    Args:
        export_results: Dictionary of export results by format
        document_set: Original document set used for export
        output_report_path: Optional path to save the consistency report
        
    Returns:
        ConsistencyReport with detailed analysis
    """
    validator = FormatConsistencyValidator()
    report = validator.validate_consistency(export_results, document_set)
    
    if output_report_path:
        validator.generate_consistency_report(report, output_report_path)
    
    return report