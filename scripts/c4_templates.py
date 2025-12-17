"""
C4 Model diagram templates and styling for Mermaid diagrams.

This module provides consistent templates and styling for generating
C4 Model architecture diagrams using Mermaid syntax.
"""

from typing import Dict, List, Any
from pathlib import Path


class C4DiagramTemplates:
    """Templates and styling for C4 Model diagrams."""
    
    # Color themes for different C4 levels
    THEMES = {
        'system_context': {
            'primaryColor': '#ff6b6b',
            'primaryTextColor': '#fff', 
            'primaryBorderColor': '#ff4757',
            'lineColor': '#5f27cd',
            'secondaryColor': '#00d2d3',
            'tertiaryColor': '#ff9ff3'
        },
        'container': {
            'primaryColor': '#3742fa',
            'primaryTextColor': '#fff',
            'primaryBorderColor': '#2f3542', 
            'lineColor': '#57606f',
            'secondaryColor': '#2ed573',
            'tertiaryColor': '#ffa502'
        },
        'component': {
            'primaryColor': '#1dd1a1',
            'primaryTextColor': '#fff',
            'primaryBorderColor': '#10ac84',
            'lineColor': '#576574', 
            'secondaryColor': '#feca57',
            'tertiaryColor': '#ff6348'
        },
        'code': {
            'primaryColor': '#8e44ad',
            'primaryTextColor': '#fff',
            'primaryBorderColor': '#6c3483',
            'lineColor': '#34495e',
            'secondaryColor': '#e67e22', 
            'tertiaryColor': '#2ecc71'
        }
    }
    
    @classmethod
    def get_theme_config(cls, diagram_type: str) -> str:
        """Get Mermaid theme configuration for diagram type."""
        theme = cls.THEMES.get(diagram_type, cls.THEMES['system_context'])
        
        theme_vars = ', '.join([f"'{k}':'{v}'" for k, v in theme.items()])
        return f"%%{{init: {{'theme':'base', 'themeVariables': {{{theme_vars}}}}}}}%%"
    
    @classmethod
    def create_system_context_template(cls, title: str, system_name: str, description: str) -> str:
        """Create system context diagram template."""
        theme_config = cls.get_theme_config('system_context')
        
        return f"""{theme_config}
C4Context
    title {title}
    
    Person_Ext(researcher, "Researcher", "Studies child development patterns using drawing analysis")
    Person_Ext(educator, "Educational Professional", "Monitors developmental milestones in children") 
    Person_Ext(healthcare, "Healthcare Provider", "Screens for developmental concerns")
    
    System(main_system, "{system_name}", "{description}")
    
"""
    
    @classmethod
    def create_container_template(cls, title: str) -> str:
        """Create container diagram template."""
        theme_config = cls.get_theme_config('container')
        
        return f"""{theme_config}
C4Container
    title {title}
    
    Person(user, "System User", "Interacts with the system")
    
"""
    
    @classmethod
    def create_component_template(cls, title: str, container_name: str = "Backend API Container") -> str:
        """Create component diagram template."""
        theme_config = cls.get_theme_config('component')
        
        return f"""{theme_config}
C4Component
    title {title}
    
    Container_Boundary(backend, "{container_name}") {{
"""
    
    @classmethod
    def create_class_diagram_template(cls, title: str = "Class Structure") -> str:
        """Create class diagram template."""
        theme_config = cls.get_theme_config('code')
        
        return f"""{theme_config}
classDiagram
    direction TB
    
"""
    
    @classmethod
    def format_external_system(cls, system_id: str, name: str, description: str) -> str:
        """Format external system element."""
        return f'    System_Ext({system_id}, "{name}", "{description}")'
    
    @classmethod
    def format_container(cls, container_id: str, name: str, technology: str, description: str) -> str:
        """Format container element."""
        return f'    Container({container_id}, "{name}", "{technology}", "{description}")'
    
    @classmethod
    def format_component(cls, comp_id: str, name: str, technology: str, description: str, indent: int = 8) -> str:
        """Format component element with proper indentation."""
        spaces = ' ' * indent
        return f'{spaces}Component({comp_id}, "{name}", "{technology}", "{description}")'
    
    @classmethod
    def format_relationship(cls, from_id: str, to_id: str, label: str, technology: str = "", indent: int = 4) -> str:
        """Format relationship between elements."""
        spaces = ' ' * indent
        tech_part = f', "{technology}"' if technology else ''
        return f'{spaces}Rel({from_id}, {to_id}, "{label}"{tech_part})'
    
    @classmethod
    def format_class_definition(cls, class_name: str, attributes: List[str], methods: List[str]) -> str:
        """Format class definition for class diagram."""
        content = f"    class {class_name} {{\n"
        
        # Add attributes
        for attr in attributes:
            visibility = '+' if not attr.startswith('_') else '-'
            content += f"        {visibility}{attr}\n"
        
        # Add methods  
        for method in methods:
            visibility = '+' if not method.startswith('_') else '-'
            content += f"        {visibility}{method}\n"
        
        content += "    }\n"
        return content
    
    @classmethod
    def format_class_relationship(cls, from_class: str, to_class: str, relationship_type: str) -> str:
        """Format class relationship."""
        relationship_symbols = {
            'inherits': '<|--',
            'implements': '<|..',
            'depends_on': '-->',
            'uses': '..>',
            'aggregates': 'o--',
            'composes': '*--'
        }
        
        symbol = relationship_symbols.get(relationship_type, '-->')
        return f"    {to_class} {symbol} {from_class}"
    
    @classmethod
    def add_responsive_styling(cls, content: str) -> str:
        """Add responsive styling directives to diagram."""
        responsive_config = """
<style>
.mermaid {
    max-width: 100%;
    height: auto;
}

@media (max-width: 768px) {
    .mermaid {
        font-size: 12px;
    }
}

@media (max-width: 480px) {
    .mermaid {
        font-size: 10px;
    }
}
</style>
"""
        return content + responsive_config
    
    @classmethod
    def create_diagram_legend(cls, diagram_type: str) -> str:
        """Create legend for diagram type."""
        legends = {
            'system_context': """
## Legend

- **Person (Blue)**: External users of the system
- **System (Red)**: The main system being documented  
- **External System (Gray)**: External dependencies
- **Relationship (Arrow)**: Interaction between elements
""",
            'container': """
## Legend

- **Person (Blue)**: System users
- **Container (Green)**: Deployable/executable units
- **Database (Yellow)**: Data storage containers
- **Relationship (Arrow)**: Communication between containers
""",
            'component': """
## Legend

- **Component (Teal)**: Internal building blocks
- **API (Orange)**: Interface components
- **Service (Purple)**: Business logic components
- **Model (Red)**: Data access components
""",
            'code': """
## Legend

- **Class (Purple)**: Code classes and interfaces
- **Inheritance (Solid arrow)**: IS-A relationships
- **Dependency (Dashed arrow)**: USES relationships
- **Composition (Diamond)**: PART-OF relationships
"""
        }
        
        return legends.get(diagram_type, "")
    
    @classmethod
    def validate_diagram_syntax(cls, mermaid_content: str) -> Dict[str, Any]:
        """Validate Mermaid diagram syntax."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        lines = mermaid_content.split('\n')
        
        # Basic syntax validation
        has_diagram_type = False
        has_title = False
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            
            # Check for diagram type
            if any(diagram_type in line for diagram_type in ['C4Context', 'C4Container', 'C4Component', 'classDiagram']):
                has_diagram_type = True
            
            # Check for title
            if line.startswith('title '):
                has_title = True
            
            # Check for common syntax errors
            if line and not line.startswith('%%') and not line.startswith('```'):
                # Check for unmatched parentheses
                if line.count('(') != line.count(')'):
                    validation_result['errors'].append(f"Line {i}: Unmatched parentheses")
                
                # Check for unmatched quotes
                if line.count('"') % 2 != 0:
                    validation_result['errors'].append(f"Line {i}: Unmatched quotes")
        
        if not has_diagram_type:
            validation_result['errors'].append("Missing diagram type declaration")
        
        if not has_title:
            validation_result['warnings'].append("Missing diagram title")
        
        validation_result['valid'] = len(validation_result['errors']) == 0
        
        return validation_result


class C4DiagramGenerator:
    """High-level generator for C4 diagrams using templates."""
    
    def __init__(self):
        self.templates = C4DiagramTemplates()
    
    def generate_system_context_diagram(self, context_data: Dict[str, Any]) -> str:
        """Generate complete system context diagram."""
        system_name = context_data.get('system_name', 'System')
        description = context_data.get('description', 'System description')
        
        # Start with template
        content = self.templates.create_system_context_template(
            f"System Context - {system_name}",
            system_name,
            description
        )
        
        # Add external systems
        for i, ext_system in enumerate(context_data.get('external_systems', [])):
            system_id = f"ext_sys_{i}"
            system_desc = self._get_external_system_description(ext_system)
            content += self.templates.format_external_system(system_id, ext_system, system_desc) + '\n'
        
        # Add relationships
        content += '\n'
        content += self.templates.format_relationship('researcher', 'main_system', 'Uploads drawings, views analysis', 'HTTPS') + '\n'
        content += self.templates.format_relationship('educator', 'main_system', 'Monitors progress, generates reports', 'HTTPS') + '\n'
        content += self.templates.format_relationship('healthcare', 'main_system', 'Screens patients, exports findings', 'HTTPS') + '\n'
        
        # Add external system relationships
        for i, ext_system in enumerate(context_data.get('external_systems', [])):
            system_id = f"ext_sys_{i}"
            rel_desc, protocol = self._get_external_system_relationship(ext_system)
            content += self.templates.format_relationship('main_system', system_id, rel_desc, protocol) + '\n'
        
        return content
    
    def _get_external_system_description(self, system_name: str) -> str:
        """Get description for external system."""
        descriptions = {
            "File Storage System": "Stores uploaded drawings and generated visualizations",
            "ML Model Repository": "Stores trained autoencoder models and ViT embeddings",
            "AWS Services": "Optional cloud training and deployment services",
            "Container Registry": "Docker image storage and distribution"
        }
        return descriptions.get(system_name, "External system dependency")
    
    def _get_external_system_relationship(self, system_name: str) -> tuple:
        """Get relationship description and protocol for external system."""
        relationships = {
            "File Storage System": ("Stores/retrieves files", "File I/O"),
            "ML Model Repository": ("Loads models, saves training results", "File I/O"),
            "AWS Services": ("Optional model training", "HTTPS/API"),
            "Container Registry": ("Pulls/pushes images", "HTTPS")
        }
        return relationships.get(system_name, ("Uses", "API"))