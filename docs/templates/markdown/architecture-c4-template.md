# {{DIAGRAM_TITLE}} (C4 Level {{LEVEL}})

## Overview

{{SYSTEM_DESCRIPTION}}

## {{DIAGRAM_TYPE}} Diagram

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryColor':'#ff6b6b', 'primaryTextColor':'#fff', 'primaryBorderColor':'#ff4757', 'lineColor':'#5f27cd', 'secondaryColor':'#00d2d3', 'tertiaryColor':'#ff9ff3'}}}%%
{{DIAGRAM_TYPE}}
    title {{DIAGRAM_TITLE}}
    
    {{DIAGRAM_CONTENT}}
```

## Legend

- **Person (Blue)**: External users of the system
- **System (Red)**: The main system being documented  
- **External System (Gray)**: External dependencies
- **Container (Light Blue)**: Applications and data stores
- **Component (Green)**: Individual components within containers
- **Relationship (Arrow)**: Interaction between elements

## Diagram Validation

{{VALIDATION_NOTES}}

## {{ELEMENT_TYPE}} Details

{{#ELEMENTS}}
### {{ELEMENT_NAME}}
{{ELEMENT_DESCRIPTION}}

{{#ELEMENT_PROPERTIES}}
- **{{PROPERTY_NAME}}**: {{PROPERTY_VALUE}}
{{/ELEMENT_PROPERTIES}}

{{/ELEMENTS}}

## Relationships

{{#RELATIONSHIPS}}
### {{SOURCE}} → {{TARGET}}
- **Description**: {{RELATIONSHIP_DESCRIPTION}}
- **Protocol**: {{PROTOCOL}}
- **Data Flow**: {{DATA_FLOW}}
{{/RELATIONSHIPS}}

## Architecture Decisions

{{#DECISIONS}}
### {{DECISION_TITLE}}
- **Status**: {{STATUS}}
- **Context**: {{CONTEXT}}
- **Decision**: {{DECISION}}
- **Consequences**: {{CONSEQUENCES}}
{{/DECISIONS}}

## Quality Attributes

### Performance
{{PERFORMANCE_REQUIREMENTS}}

### Security
{{SECURITY_REQUIREMENTS}}

### Scalability
{{SCALABILITY_REQUIREMENTS}}

### Maintainability
{{MAINTAINABILITY_REQUIREMENTS}}

---

**Generated**: {{GENERATION_DATE}}  
**Version**: {{VERSION}}  
**Last Updated**: {{LAST_UPDATED}}  
**Validated**: {{VALIDATION_STATUS}}