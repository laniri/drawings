#!/usr/bin/env python3
"""
LaTeX Rendering System

This module provides LaTeX mathematical notation rendering for algorithm documentation,
with support for HTML (MathJax), PDF, and other export formats.

**Requirements: 4.4, 4.5**
"""

import re
import html
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LaTeXExpression:
    """Represents a LaTeX mathematical expression."""
    original_text: str
    latex_code: str
    display_mode: bool = False  # True for display math, False for inline
    description: str = ""


class LaTeXConverter:
    """Converts mathematical notation to LaTeX format."""
    
    def __init__(self):
        # Unicode to LaTeX symbol mappings
        self.symbol_mappings = {
            # Greek letters (lowercase)
            'α': r'\alpha',
            'β': r'\beta', 
            'γ': r'\gamma',
            'δ': r'\delta',
            'ε': r'\epsilon',
            'ζ': r'\zeta',
            'η': r'\eta',
            'θ': r'\theta',
            'ι': r'\iota',
            'κ': r'\kappa',
            'λ': r'\lambda',
            'μ': r'\mu',
            'ν': r'\nu',
            'ξ': r'\xi',
            'ο': r'o',
            'π': r'\pi',
            'ρ': r'\rho',
            'σ': r'\sigma',
            'τ': r'\tau',
            'υ': r'\upsilon',
            'φ': r'\phi',
            'χ': r'\chi',
            'ψ': r'\psi',
            'ω': r'\omega',
            
            # Greek letters (uppercase)
            'Α': r'A',
            'Β': r'B',
            'Γ': r'\Gamma',
            'Δ': r'\Delta',
            'Ε': r'E',
            'Ζ': r'Z',
            'Η': r'H',
            'Θ': r'\Theta',
            'Ι': r'I',
            'Κ': r'K',
            'Λ': r'\Lambda',
            'Μ': r'M',
            'Ν': r'N',
            'Ξ': r'\Xi',
            'Ο': r'O',
            'Π': r'\Pi',
            'Ρ': r'P',
            'Σ': r'\Sigma',
            'Τ': r'T',
            'Υ': r'\Upsilon',
            'Φ': r'\Phi',
            'Χ': r'X',
            'Ψ': r'\Psi',
            'Ω': r'\Omega',
            
            # Mathematical operators
            '∑': r'\sum',
            '∏': r'\prod',
            '∫': r'\int',
            '∮': r'\oint',
            '∂': r'\partial',
            '∇': r'\nabla',
            '∆': r'\Delta',
            '√': r'\sqrt',
            '∞': r'\infty',
            
            # Relations
            '≤': r'\leq',
            '≥': r'\geq',
            '≠': r'\neq',
            '≈': r'\approx',
            '≡': r'\equiv',
            '∝': r'\propto',
            '∼': r'\sim',
            '≃': r'\simeq',
            '≅': r'\cong',
            '∈': r'\in',
            '∉': r'\notin',
            '∋': r'\ni',
            '⊂': r'\subset',
            '⊃': r'\supset',
            '⊆': r'\subseteq',
            '⊇': r'\supseteq',
            '∪': r'\cup',
            '∩': r'\cap',
            '∅': r'\emptyset',
            
            # Arrows
            '→': r'\rightarrow',
            '←': r'\leftarrow',
            '↔': r'\leftrightarrow',
            '⇒': r'\Rightarrow',
            '⇐': r'\Leftarrow',
            '⇔': r'\Leftrightarrow',
            '↑': r'\uparrow',
            '↓': r'\downarrow',
            
            # Set notation
            'ℝ': r'\mathbb{R}',
            'ℕ': r'\mathbb{N}',
            'ℤ': r'\mathbb{Z}',
            'ℚ': r'\mathbb{Q}',
            'ℂ': r'\mathbb{C}',
            'ℙ': r'\mathbb{P}',
            
            # Other symbols
            '±': r'\pm',
            '∓': r'\mp',
            '×': r'\times',
            '÷': r'\div',
            '·': r'\cdot',
            '∘': r'\circ',
            '†': r'\dagger',
            '‡': r'\ddagger',
            '℘': r'\wp',
            'ℜ': r'\Re',
            'ℑ': r'\Im',
            '∀': r'\forall',
            '∃': r'\exists',
            '∄': r'\nexists',
            '∧': r'\land',
            '∨': r'\lor',
            '¬': r'\lnot',
            '⊕': r'\oplus',
            '⊗': r'\otimes',
            '⊥': r'\perp',
            '∥': r'\parallel',
            
            # Subscripts and superscripts (common ones)
            '₀': r'_0', '₁': r'_1', '₂': r'_2', '₃': r'_3', '₄': r'_4',
            '₅': r'_5', '₆': r'_6', '₇': r'_7', '₈': r'_8', '₉': r'_9',
            '⁰': r'^0', '¹': r'^1', '²': r'^2', '³': r'^3', '⁴': r'^4',
            '⁵': r'^5', '⁶': r'^6', '⁷': r'^7', '⁸': r'^8', '⁹': r'^9',
            
            # Special characters
            '°': r'^\circ',
            '′': r"'",
            '″': r"''",
            '‴': r"'''",
        }
        
        # Patterns for mathematical expressions
        self.math_patterns = {
            # Function notation: f(x), g(x,y), etc.
            'function': re.compile(r'([a-zA-Z])\(([^)]+)\)'),
            
            # Fractions: a/b
            'fraction': re.compile(r'(\w+|\([^)]+\))/(\w+|\([^)]+\))'),
            
            # Powers: x^2, x^{n+1}
            'power': re.compile(r'(\w+|\([^)]+\))\^(\w+|\{[^}]+\})'),
            
            # Subscripts: x_i, x_{i+1}
            'subscript': re.compile(r'(\w+)_(\w+|\{[^}]+\})'),
            
            # Square roots: sqrt(x)
            'sqrt': re.compile(r'sqrt\(([^)]+)\)'),
            
            # Summation: sum_{i=1}^n or Σ_{i=1}^n
            'summation': re.compile(r'(?:sum|Σ)_\{([^}]+)\}\^?\{?([^}]*)\}?'),
            
            # Product: prod_{i=1}^n or Π_{i=1}^n
            'product': re.compile(r'(?:prod|Π)_\{([^}]+)\}\^?\{?([^}]*)\}?'),
            
            # Integral: int_a^b or ∫_a^b
            'integral': re.compile(r'(?:int|∫)_([^\\s]+)\^([^\\s]+)'),
            
            # Limits: lim_{x->0}
            'limit': re.compile(r'lim_\{([^}]+)\}'),
            
            # Matrix notation: [a b; c d]
            'matrix': re.compile(r'\[([^]]+)\]'),
        }
    
    def convert_to_latex(self, text: str) -> str:
        """Convert mathematical text to LaTeX format."""
        latex_text = text
        
        # First, convert Unicode symbols
        for unicode_char, latex_symbol in self.symbol_mappings.items():
            latex_text = latex_text.replace(unicode_char, latex_symbol)
        
        # Apply mathematical pattern conversions
        latex_text = self._convert_functions(latex_text)
        latex_text = self._convert_fractions(latex_text)
        latex_text = self._convert_powers(latex_text)
        latex_text = self._convert_subscripts(latex_text)
        latex_text = self._convert_sqrt(latex_text)
        latex_text = self._convert_summations(latex_text)
        latex_text = self._convert_products(latex_text)
        latex_text = self._convert_integrals(latex_text)
        latex_text = self._convert_limits(latex_text)
        
        return latex_text
    
    def _convert_functions(self, text: str) -> str:
        """Convert function notation to LaTeX."""
        def replace_func(match):
            func_name = match.group(1)
            args = match.group(2)
            return f"{func_name}({args})"
        
        return self.math_patterns['function'].sub(replace_func, text)
    
    def _convert_fractions(self, text: str) -> str:
        """Convert fractions to LaTeX \\frac format."""
        def replace_frac(match):
            numerator = match.group(1)
            denominator = match.group(2)
            return f"\\frac{{{numerator}}}{{{denominator}}}"
        
        return self.math_patterns['fraction'].sub(replace_frac, text)
    
    def _convert_powers(self, text: str) -> str:
        """Convert powers to LaTeX format."""
        def replace_power(match):
            base = match.group(1)
            exponent = match.group(2)
            if exponent.startswith('{') and exponent.endswith('}'):
                return f"{base}^{exponent}"
            else:
                return f"{base}^{{{exponent}}}"
        
        return self.math_patterns['power'].sub(replace_power, text)
    
    def _convert_subscripts(self, text: str) -> str:
        """Convert subscripts to LaTeX format."""
        def replace_subscript(match):
            base = match.group(1)
            subscript = match.group(2)
            if subscript.startswith('{') and subscript.endswith('}'):
                return f"{base}_{subscript}"
            else:
                return f"{base}_{{{subscript}}}"
        
        return self.math_patterns['subscript'].sub(replace_subscript, text)
    
    def _convert_sqrt(self, text: str) -> str:
        """Convert square root notation to LaTeX."""
        def replace_sqrt(match):
            content = match.group(1)
            return f"\\sqrt{{{content}}}"
        
        return self.math_patterns['sqrt'].sub(replace_sqrt, text)
    
    def _convert_summations(self, text: str) -> str:
        """Convert summation notation to LaTeX."""
        def replace_sum(match):
            lower = match.group(1)
            upper = match.group(2)
            if upper:
                return f"\\sum_{{{lower}}}^{{{upper}}}"
            else:
                return f"\\sum_{{{lower}}}"
        
        return self.math_patterns['summation'].sub(replace_sum, text)
    
    def _convert_products(self, text: str) -> str:
        """Convert product notation to LaTeX."""
        def replace_prod(match):
            lower = match.group(1)
            upper = match.group(2)
            if upper:
                return f"\\prod_{{{lower}}}^{{{upper}}}"
            else:
                return f"\\prod_{{{lower}}}"
        
        return self.math_patterns['product'].sub(replace_prod, text)
    
    def _convert_integrals(self, text: str) -> str:
        """Convert integral notation to LaTeX."""
        def replace_int(match):
            lower = match.group(1)
            upper = match.group(2)
            return f"\\int_{{{lower}}}^{{{upper}}}"
        
        return self.math_patterns['integral'].sub(replace_int, text)
    
    def _convert_limits(self, text: str) -> str:
        """Convert limit notation to LaTeX."""
        def replace_limit(match):
            condition = match.group(1)
            return f"\\lim_{{{condition}}}"
        
        return self.math_patterns['limit'].sub(replace_limit, text)
    
    def extract_math_expressions(self, text: str) -> List[LaTeXExpression]:
        """Extract mathematical expressions from text."""
        expressions = []
        
        # Look for display math (equations on their own lines)
        display_math_pattern = re.compile(r'^[ \t]*([^a-zA-Z]*(?:' + '|'.join(self.symbol_mappings.keys()) + r')[^a-zA-Z]*?)[ \t]*$', re.MULTILINE)
        
        for match in display_math_pattern.finditer(text):
            original = match.group(1).strip()
            if original and len(original) > 2:  # Skip very short matches
                latex_code = self.convert_to_latex(original)
                expressions.append(LaTeXExpression(
                    original_text=original,
                    latex_code=latex_code,
                    display_mode=True
                ))
        
        # Look for inline math (mathematical symbols within text)
        inline_math_pattern = re.compile(r'([^a-zA-Z]*(?:' + '|'.join(self.symbol_mappings.keys()) + r')[^a-zA-Z]*)')
        
        for match in inline_math_pattern.finditer(text):
            original = match.group(1).strip()
            if original and len(original) <= 20:  # Inline math should be shorter
                latex_code = self.convert_to_latex(original)
                expressions.append(LaTeXExpression(
                    original_text=original,
                    latex_code=latex_code,
                    display_mode=False
                ))
        
        return expressions


class LaTeXRenderer:
    """
    Renders LaTeX mathematical notation for different output formats.
    
    **Requirements: 4.4, 4.5**
    """
    
    def __init__(self):
        self.converter = LaTeXConverter()
    
    def render_for_html(self, latex_expressions: List[LaTeXExpression]) -> str:
        """Render LaTeX expressions for HTML output with MathJax."""
        html_content = []
        
        # Add MathJax configuration
        html_content.append(self._get_mathjax_config())
        
        for expr in latex_expressions:
            if expr.display_mode:
                # Display math (block)
                html_content.append(f"$$\n{expr.latex_code}\n$$")
            else:
                # Inline math
                html_content.append(f"\\({expr.latex_code}\\)")
        
        return '\n'.join(html_content)
    
    def render_for_markdown(self, text: str) -> str:
        """Render mathematical expressions in markdown format."""
        expressions = self.converter.extract_math_expressions(text)
        rendered_text = text
        
        # Replace expressions with markdown math notation
        for expr in expressions:
            if expr.display_mode:
                # Display math
                math_block = f"\n$$\n{expr.latex_code}\n$$\n"
                rendered_text = rendered_text.replace(expr.original_text, math_block)
            else:
                # Inline math
                inline_math = f"${expr.latex_code}$"
                rendered_text = rendered_text.replace(expr.original_text, inline_math)
        
        return rendered_text
    
    def render_for_pdf(self, latex_expressions: List[LaTeXExpression]) -> str:
        """Render LaTeX expressions for PDF output."""
        latex_content = []
        
        # Add LaTeX document preamble
        latex_content.append(self._get_latex_preamble())
        
        for expr in latex_expressions:
            if expr.display_mode:
                # Display math
                latex_content.append(f"\\[\n{expr.latex_code}\n\\]")
            else:
                # Inline math
                latex_content.append(f"${expr.latex_code}$")
        
        return '\n'.join(latex_content)
    
    def render_algorithm_documentation(self, algorithm_content: str) -> Dict[str, str]:
        """Render algorithm documentation with mathematical notation for multiple formats."""
        expressions = self.converter.extract_math_expressions(algorithm_content)
        
        rendered_formats = {
            'html': self._render_content_for_html(algorithm_content, expressions),
            'markdown': self.render_for_markdown(algorithm_content),
            'latex': self._render_content_for_latex(algorithm_content, expressions),
            'plain': self._render_content_plain(algorithm_content, expressions)
        }
        
        return rendered_formats
    
    def _get_mathjax_config(self) -> str:
        """Get MathJax configuration for HTML rendering."""
        return '''
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.0/es5/tex-mml-chtml.js">
</script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [['\\\\(','\\\\)']],
    displayMath: [['$$','$$']],
    processEscapes: true,
    processEnvironments: true
  },
  displayAlign: "center",
  CommonHTML: { linebreaks: { automatic: true } },
  "HTML-CSS": { linebreaks: { automatic: true } }
});
</script>
'''
    
    def _get_latex_preamble(self) -> str:
        """Get LaTeX document preamble."""
        return '''
\\documentclass{article}
\\usepackage{amsmath}
\\usepackage{amsfonts}
\\usepackage{amssymb}
\\usepackage{mathtools}
\\usepackage{unicode-math}
\\begin{document}
'''
    
    def _render_content_for_html(self, content: str, expressions: List[LaTeXExpression]) -> str:
        """Render content for HTML with MathJax."""
        html_content = content
        
        # Replace mathematical expressions with MathJax notation
        for expr in expressions:
            if expr.display_mode:
                math_html = f"<div class='math-display'>$$\n{expr.latex_code}\n$$</div>"
            else:
                math_html = f"<span class='math-inline'>\\({expr.latex_code}\\)</span>"
            
            html_content = html_content.replace(expr.original_text, math_html)
        
        # Add MathJax configuration
        full_html = self._get_mathjax_config() + "\n" + html_content
        
        return full_html
    
    def _render_content_for_latex(self, content: str, expressions: List[LaTeXExpression]) -> str:
        """Render content for LaTeX/PDF output."""
        latex_content = content
        
        # Replace mathematical expressions with LaTeX notation
        for expr in expressions:
            if expr.display_mode:
                latex_math = f"\\[\n{expr.latex_code}\n\\]"
            else:
                latex_math = f"${expr.latex_code}$"
            
            latex_content = latex_content.replace(expr.original_text, latex_math)
        
        return latex_content
    
    def _render_content_plain(self, content: str, expressions: List[LaTeXExpression]) -> str:
        """Render content in plain text format (fallback)."""
        plain_content = content
        
        # Keep original mathematical notation for plain text
        # This serves as a fallback when LaTeX rendering is not available
        return plain_content
    
    def generate_math_css(self) -> str:
        """Generate CSS styles for mathematical content."""
        return '''
/* Mathematical notation styles */
.math-display {
    text-align: center;
    margin: 1em 0;
    padding: 0.5em;
    background-color: #f8f9fa;
    border-left: 4px solid #007bff;
    border-radius: 4px;
}

.math-inline {
    font-family: 'Times New Roman', serif;
    font-style: italic;
}

.algorithm-section {
    margin: 2em 0;
}

.algorithm-section h3 {
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 0.5em;
}

.complexity-analysis {
    background-color: #f1f8ff;
    border: 1px solid #c0d3eb;
    border-radius: 6px;
    padding: 1em;
    margin: 1em 0;
}

.performance-metrics {
    background-color: #f6f8fa;
    border: 1px solid #d1d5da;
    border-radius: 6px;
    padding: 1em;
    margin: 1em 0;
}

.mathematical-formulation {
    background-color: #fffbf0;
    border: 1px solid #f1c40f;
    border-radius: 6px;
    padding: 1em;
    margin: 1em 0;
}

.latex-equation {
    font-family: 'Computer Modern', 'Times New Roman', serif;
    font-size: 1.1em;
    line-height: 1.6;
}
'''
    
    def validate_latex_syntax(self, latex_code: str) -> Tuple[bool, List[str]]:
        """Validate LaTeX syntax and return errors if any."""
        errors = []
        
        # Check for balanced braces
        brace_count = 0
        for char in latex_code:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count < 0:
                    errors.append("Unmatched closing brace '}' found")
                    break
        
        if brace_count > 0:
            errors.append(f"Unmatched opening braces: {brace_count} '{' without closing '}'")
        
        # Check for common LaTeX command errors
        invalid_commands = re.findall(r'\\([a-zA-Z]+)', latex_code)
        known_commands = {
            'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
            'lambda', 'mu', 'nu', 'xi', 'pi', 'rho', 'sigma', 'tau', 'phi', 'chi', 'psi', 'omega',
            'sum', 'prod', 'int', 'frac', 'sqrt', 'lim', 'infty', 'partial', 'nabla',
            'leq', 'geq', 'neq', 'approx', 'in', 'notin', 'subset', 'cup', 'cap',
            'mathbb', 'text', 'left', 'right', 'begin', 'end'
        }
        
        for cmd in invalid_commands:
            if cmd not in known_commands:
                errors.append(f"Unknown LaTeX command: \\{cmd}")
        
        return len(errors) == 0, errors


def main():
    """Main entry point for LaTeX rendering system."""
    renderer = LaTeXRenderer()
    
    # Example mathematical content
    test_content = """
    Algorithm Implementation with Mathematical Formulation
    
    Core equation: f(x) = Σᵢ₌₁ⁿ wᵢ * xᵢ + b
    
    Where:
    - wᵢ ∈ ℝ are learned parameters
    - xᵢ ∈ ℝⁿ are input features  
    - b ∈ ℝ is the bias term
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Optimization objective: min_w L(w) = (1/m) Σⱼ₌₁ᵐ ℓ(f(xⱼ), yⱼ) + λR(w)
    """
    
    # Render for different formats
    rendered_formats = renderer.render_algorithm_documentation(test_content)
    
    print("✅ LaTeX rendering system test completed!")
    print("\n📝 Markdown format:")
    print(rendered_formats['markdown'][:200] + "...")
    
    print("\n🌐 HTML format (sample):")
    print(rendered_formats['html'][:200] + "...")
    
    print("\n📄 LaTeX format (sample):")
    print(rendered_formats['latex'][:200] + "...")


if __name__ == "__main__":
    main()