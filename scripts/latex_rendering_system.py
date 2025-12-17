#!/usr/bin/env python3
"""
LaTeX Rendering System

This module provides LaTeX mathematical notation rendering for all export formats,
MathJax integration for HTML output, and PDF-compatible mathematical notation rendering.
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RenderFormat(Enum):
    """Supported rendering formats."""
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    LATEX = "latex"


@dataclass
class MathJaxConfig:
    """Configuration for MathJax rendering."""
    version: str = "3.2.2"
    extensions: List[str] = None
    tex_options: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extensions is None:
            self.extensions = ['tex2jax', 'TeX', 'AMSmath', 'AMSsymbols']
        
        if self.tex_options is None:
            self.tex_options = {
                'inlineMath': [['$', '$'], ['\\(', '\\)']],
                'displayMath': [['$$', '$$'], ['\\[', '\\]']],
                'processEscapes': True,
                'processEnvironments': True
            }


@dataclass
class LaTeXDocument:
    """Represents a LaTeX document structure."""
    title: str
    content: str
    packages: List[str] = None
    document_class: str = "article"
    
    def __post_init__(self):
        if self.packages is None:
            self.packages = ['amsmath', 'amssymb', 'amsfonts', 'mathtools']


class LaTeXRenderer:
    """
    LaTeX mathematical notation renderer with multi-format support.
    
    Provides LaTeX mathematical notation rendering for all export formats,
    MathJax integration for HTML output, and PDF-compatible rendering.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.mathjax_config = MathJaxConfig()
        
        # Mathematical symbol mappings
        self.symbol_mappings = {
            # Greek letters (lowercase)
            'α': r'\alpha', 'β': r'\beta', 'γ': r'\gamma', 'δ': r'\delta',
            'ε': r'\epsilon', 'ζ': r'\zeta', 'η': r'\eta', 'θ': r'\theta',
            'ι': r'\iota', 'κ': r'\kappa', 'λ': r'\lambda', 'μ': r'\mu',
            'ν': r'\nu', 'ξ': r'\xi', 'ο': r'\omicron', 'π': r'\pi',
            'ρ': r'\rho', 'σ': r'\sigma', 'τ': r'\tau', 'υ': r'\upsilon',
            'φ': r'\phi', 'χ': r'\chi', 'ψ': r'\psi', 'ω': r'\omega',
            
            # Greek letters (uppercase)
            'Α': r'\Alpha', 'Β': r'\Beta', 'Γ': r'\Gamma', 'Δ': r'\Delta',
            'Ε': r'\Epsilon', 'Ζ': r'\Zeta', 'Η': r'\Eta', 'Θ': r'\Theta',
            'Ι': r'\Iota', 'Κ': r'\Kappa', 'Λ': r'\Lambda', 'Μ': r'\Mu',
            'Ν': r'\Nu', 'Ξ': r'\Xi', 'Ο': r'\Omicron', 'Π': r'\Pi',
            'Ρ': r'\Rho', 'Σ': r'\Sigma', 'Τ': r'\Tau', 'Υ': r'\Upsilon',
            'Φ': r'\Phi', 'Χ': r'\Chi', 'Ψ': r'\Psi', 'Ω': r'\Omega',
            
            # Mathematical operators
            '∑': r'\sum', '∏': r'\prod', '∫': r'\int', '∮': r'\oint',
            '∂': r'\partial', '∇': r'\nabla', '∆': r'\Delta', '∞': r'\infty',
            
            # Relations and comparisons
            '≤': r'\leq', '≥': r'\geq', '≠': r'\neq', '≈': r'\approx',
            '≡': r'\equiv', '∝': r'\propto', '∼': r'\sim', '≃': r'\simeq',
            
            # Set theory
            '∈': r'\in', '∉': r'\notin', '⊂': r'\subset', '⊃': r'\supset',
            '⊆': r'\subseteq', '⊇': r'\supseteq', '∪': r'\cup', '∩': r'\cap',
            '∅': r'\emptyset', '∀': r'\forall', '∃': r'\exists',
            
            # Logic
            '∧': r'\land', '∨': r'\lor', '¬': r'\neg', '→': r'\rightarrow',
            '←': r'\leftarrow', '↔': r'\leftrightarrow', '⇒': r'\Rightarrow',
            '⇐': r'\Leftarrow', '⇔': r'\Leftrightarrow',
            
            # Number sets
            'ℝ': r'\mathbb{R}', 'ℕ': r'\mathbb{N}', 'ℤ': r'\mathbb{Z}',
            'ℚ': r'\mathbb{Q}', 'ℂ': r'\mathbb{C}', 'ℙ': r'\mathbb{P}',
            
            # Other symbols
            '±': r'\pm', '∓': r'\mp', '×': r'\times', '÷': r'\div',
            '√': r'\sqrt', '∝': r'\propto', '°': r'^\circ'
        }
        
        # LaTeX environments and commands
        self.latex_environments = {
            'equation': r'\begin{equation}',
            'align': r'\begin{align}',
            'matrix': r'\begin{matrix}',
            'cases': r'\begin{cases}'
        }
    
    def convert_unicode_to_latex(self, text: str) -> str:
        """
        Convert Unicode mathematical symbols to LaTeX commands.
        
        Args:
            text: Text containing Unicode mathematical symbols
            
        Returns:
            Text with LaTeX commands
        """
        latex_text = text
        
        # Replace Unicode symbols with LaTeX commands
        for unicode_symbol, latex_command in self.symbol_mappings.items():
            latex_text = latex_text.replace(unicode_symbol, latex_command)
        
        # Handle subscripts and superscripts
        latex_text = self._convert_subscripts_superscripts(latex_text)
        
        # Handle fractions
        latex_text = self._convert_fractions(latex_text)
        
        # Handle square roots
        latex_text = self._convert_square_roots(latex_text)
        
        return latex_text
    
    def _convert_subscripts_superscripts(self, text: str) -> str:
        """Convert subscripts and superscripts to LaTeX format."""
        # Convert subscripts (e.g., x_i -> x_{i}, x_123 -> x_{123})
        text = re.sub(r'([a-zA-Z])_([a-zA-Z0-9]+)', r'\1_{\2}', text)
        
        # Convert superscripts (e.g., x^2 -> x^{2}, x^abc -> x^{abc})
        text = re.sub(r'([a-zA-Z])\^([a-zA-Z0-9]+)', r'\1^{\2}', text)
        
        # Handle more complex subscripts/superscripts with parentheses
        text = re.sub(r'([a-zA-Z])_\(([^)]+)\)', r'\1_{\2}', text)
        text = re.sub(r'([a-zA-Z])\^\(([^)]+)\)', r'\1^{\2}', text)
        
        return text
    
    def _convert_fractions(self, text: str) -> str:
        """Convert fraction notation to LaTeX format."""
        # Simple fractions like 1/2, 3/4
        text = re.sub(r'(\d+)/(\d+)', r'\\frac{\1}{\2}', text)
        
        # More complex fractions with parentheses
        text = re.sub(r'\(([^)]+)\)/\(([^)]+)\)', r'\\frac{\1}{\2}', text)
        
        # Variables in fractions
        text = re.sub(r'([a-zA-Z]+)/([a-zA-Z]+)', r'\\frac{\1}{\2}', text)
        
        return text
    
    def _convert_square_roots(self, text: str) -> str:
        """Convert square root notation to LaTeX format."""
        # sqrt(x) -> \sqrt{x}
        text = re.sub(r'sqrt\(([^)]+)\)', r'\\sqrt{\1}', text)
        
        # √x -> \sqrt{x} (for single variables)
        text = re.sub(r'√([a-zA-Z0-9]+)', r'\\sqrt{\1}', text)
        
        return text
    
    def render_for_html(self, latex_content: str, inline: bool = False) -> str:
        """
        Render LaTeX content for HTML output with MathJax.
        
        Args:
            latex_content: LaTeX mathematical content
            inline: Whether to render as inline math
            
        Returns:
            HTML with MathJax-compatible LaTeX
        """
        # Convert Unicode symbols to LaTeX
        latex_content = self.convert_unicode_to_latex(latex_content)
        
        # Wrap in appropriate delimiters for MathJax
        if inline:
            return f"${latex_content}$"
        else:
            return f"$${latex_content}$$"
    
    def render_for_pdf(self, latex_content: str) -> str:
        """
        Render LaTeX content for PDF output.
        
        Args:
            latex_content: LaTeX mathematical content
            
        Returns:
            PDF-compatible LaTeX
        """
        # Convert Unicode symbols to LaTeX
        latex_content = self.convert_unicode_to_latex(latex_content)
        
        # Ensure proper LaTeX formatting for PDF
        if not latex_content.strip().startswith('\\'):
            # Wrap in equation environment if not already in LaTeX environment
            return f"\\begin{{equation}}\n{latex_content}\n\\end{{equation}}"
        
        return latex_content
    
    def render_for_markdown(self, latex_content: str, inline: bool = False) -> str:
        """
        Render LaTeX content for Markdown output.
        
        Args:
            latex_content: LaTeX mathematical content
            inline: Whether to render as inline math
            
        Returns:
            Markdown-compatible LaTeX
        """
        # Convert Unicode symbols to LaTeX
        latex_content = self.convert_unicode_to_latex(latex_content)
        
        # Use standard Markdown math delimiters
        if inline:
            return f"${latex_content}$"
        else:
            return f"```math\n{latex_content}\n```"
    
    def generate_mathjax_config(self) -> str:
        """
        Generate MathJax configuration for HTML documents.
        
        Returns:
            JavaScript configuration for MathJax
        """
        config = {
            'tex': self.mathjax_config.tex_options,
            'svg': {
                'fontCache': 'global'
            },
            'startup': {
                'ready': '() => { MathJax.startup.defaultReady(); console.log("MathJax loaded"); }'
            }
        }
        
        return f"""
<script>
window.MathJax = {json.dumps(config, indent=2)};
</script>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@{self.mathjax_config.version}/es5/tex-mml-chtml.js"></script>
"""
    
    def create_latex_document(self, title: str, content: str, 
                            additional_packages: List[str] = None) -> LaTeXDocument:
        """
        Create a complete LaTeX document.
        
        Args:
            title: Document title
            content: LaTeX content
            additional_packages: Additional LaTeX packages to include
            
        Returns:
            LaTeXDocument object
        """
        packages = ['amsmath', 'amssymb', 'amsfonts', 'mathtools', 'geometry']
        
        if additional_packages:
            packages.extend(additional_packages)
        
        return LaTeXDocument(
            title=title,
            content=content,
            packages=packages
        )
    
    def render_latex_document(self, doc: LaTeXDocument) -> str:
        """
        Render a complete LaTeX document.
        
        Args:
            doc: LaTeX document to render
            
        Returns:
            Complete LaTeX document string
        """
        latex_content = []
        
        # Document class
        latex_content.append(f"\\documentclass{{{doc.document_class}}}")
        latex_content.append("")
        
        # Packages
        for package in doc.packages:
            latex_content.append(f"\\usepackage{{{package}}}")
        latex_content.append("")
        
        # Title
        latex_content.append(f"\\title{{{doc.title}}}")
        latex_content.append("\\author{Algorithm Documentation Generator}")
        latex_content.append(f"\\date{{\\today}}")
        latex_content.append("")
        
        # Document body
        latex_content.append("\\begin{document}")
        latex_content.append("")
        latex_content.append("\\maketitle")
        latex_content.append("")
        latex_content.append(doc.content)
        latex_content.append("")
        latex_content.append("\\end{document}")
        
        return '\n'.join(latex_content)
    
    def extract_math_from_markdown(self, markdown_content: str) -> List[Tuple[str, bool]]:
        """
        Extract mathematical content from Markdown.
        
        Args:
            markdown_content: Markdown content with math
            
        Returns:
            List of tuples (math_content, is_inline)
        """
        math_blocks = []
        
        # Extract display math ($$...$$)
        display_math = re.findall(r'\$\$(.*?)\$\$', markdown_content, re.DOTALL)
        for math in display_math:
            math_blocks.append((math.strip(), False))
        
        # Extract inline math ($...$)
        inline_math = re.findall(r'(?<!\$)\$([^$]+)\$(?!\$)', markdown_content)
        for math in inline_math:
            math_blocks.append((math.strip(), True))
        
        # Extract code blocks marked as math
        code_math = re.findall(r'```math\n(.*?)\n```', markdown_content, re.DOTALL)
        for math in code_math:
            math_blocks.append((math.strip(), False))
        
        return math_blocks
    
    def render_math_in_content(self, content: str, format_type: RenderFormat) -> str:
        """
        Render all mathematical content in a document for the specified format.
        
        Args:
            content: Document content with mathematical notation
            format_type: Target rendering format
            
        Returns:
            Content with rendered mathematical notation
        """
        if format_type == RenderFormat.HTML:
            return self._render_math_for_html(content)
        elif format_type == RenderFormat.PDF:
            return self._render_math_for_pdf(content)
        elif format_type == RenderFormat.MARKDOWN:
            return self._render_math_for_markdown(content)
        elif format_type == RenderFormat.LATEX:
            return self.convert_unicode_to_latex(content)
        else:
            return content
    
    def _render_math_for_html(self, content: str) -> str:
        """Render mathematical content for HTML output."""
        # Convert Unicode symbols to LaTeX
        content = self.convert_unicode_to_latex(content)
        
        # Ensure math blocks are properly delimited for MathJax
        # Replace ```math blocks with $$ delimiters
        content = re.sub(r'```math\n(.*?)\n```', r'$$\1$$', content, flags=re.DOTALL)
        
        return content
    
    def _render_math_for_pdf(self, content: str) -> str:
        """Render mathematical content for PDF output."""
        # Convert Unicode symbols to LaTeX
        content = self.convert_unicode_to_latex(content)
        
        # Convert markdown math blocks to LaTeX environments
        content = re.sub(r'```math\n(.*?)\n```', r'\\begin{equation}\n\1\n\\end{equation}', content, flags=re.DOTALL)
        content = re.sub(r'\$\$(.*?)\$\$', r'\\begin{equation}\n\1\n\\end{equation}', content, flags=re.DOTALL)
        
        return content
    
    def _render_math_for_markdown(self, content: str) -> str:
        """Render mathematical content for Markdown output."""
        # Convert Unicode symbols to LaTeX but keep Markdown math delimiters
        content = self.convert_unicode_to_latex(content)
        return content
    
    def generate_html_with_mathjax(self, title: str, content: str) -> str:
        """
        Generate complete HTML document with MathJax support.
        
        Args:
            title: Document title
            content: HTML content with mathematical notation
            
        Returns:
            Complete HTML document with MathJax
        """
        # Render math content for HTML
        rendered_content = self.render_math_in_content(content, RenderFormat.HTML)
        
        # Generate MathJax configuration
        mathjax_config = self.generate_mathjax_config()
        
        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    {mathjax_config}
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            color: #2c3e50;
            margin-top: 2em;
            margin-bottom: 1em;
        }}
        
        .math {{
            font-size: 1.1em;
        }}
        
        .equation {{
            text-align: center;
            margin: 1.5em 0;
        }}
        
        code {{
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Monaco', 'Consolas', monospace;
        }}
        
        pre {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        
        .algorithm-section {{
            border-left: 4px solid #3498db;
            padding-left: 20px;
            margin: 20px 0;
        }}
        
        .complexity-analysis {{
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }}
        
        .performance-metrics {{
            background-color: #f0f8e8;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    {rendered_content}
</body>
</html>"""
        
        return html_template
    
    def save_rendered_content(self, content: str, filename: str, format_type: RenderFormat) -> Path:
        """
        Save rendered content to file.
        
        Args:
            content: Rendered content
            filename: Output filename (without extension)
            format_type: Output format
            
        Returns:
            Path to saved file
        """
        # Determine file extension
        extensions = {
            RenderFormat.HTML: '.html',
            RenderFormat.PDF: '.tex',  # LaTeX source for PDF generation
            RenderFormat.MARKDOWN: '.md',
            RenderFormat.LATEX: '.tex'
        }
        
        extension = extensions.get(format_type, '.txt')
        output_file = self.output_dir / f"{filename}{extension}"
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save content
        output_file.write_text(content, encoding='utf-8')
        
        logger.info(f"Saved {format_type.value} content to {output_file}")
        return output_file