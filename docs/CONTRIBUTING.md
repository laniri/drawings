# Documentation Contributing Guide

## Overview

This guide explains how to contribute to and maintain the comprehensive documentation for the Children's Drawing Anomaly Detection System.

## Documentation Philosophy

Our documentation follows these principles:

1. **Industry Standards**: Use established formats (C4, OpenAPI, BPMN, UML)
2. **Automation First**: Generate documentation from code when possible
3. **Living Documentation**: Keep docs synchronized with code changes
4. **Multiple Audiences**: Serve developers, researchers, and end users
5. **Comprehensive Coverage**: Document architecture, algorithms, interfaces, and workflows

## Documentation Structure

```
docs/
├── README.md                    # Main documentation index
├── architecture/                # C4 Model architecture docs
│   ├── 01-system-context.md   # System boundaries and users
│   ├── 06-backend-components.md # Service architecture
│   └── adrs/                   # Architecture Decision Records
├── algorithms/                  # Algorithm specifications
│   ├── 07-score-normalization.md # Mathematical formulations
│   └── implementations/        # Auto-generated from code
├── api/                        # OpenAPI documentation
│   ├── README.md              # API overview
│   ├── openapi.json           # Auto-generated schema
│   └── endpoints/             # Individual endpoint docs
├── workflows/                  # BPMN process documentation
│   ├── business/              # Business process flows
│   └── technical/             # Technical workflows
├── interfaces/                 # UML interface specifications
│   ├── services/              # Service contracts
│   └── api/                   # API contracts
└── deployment/                # Infrastructure documentation
    ├── docker.md              # Container deployment
    └── environment-setup.md   # Environment configuration
```

## Documentation Types

### 1. Auto-Generated Documentation

These documents are automatically generated from code:

- **API Documentation**: Generated from FastAPI OpenAPI schema
- **Service Interfaces**: Extracted from Python docstrings
- **Database Schema**: Generated from SQLAlchemy models
- **Algorithm Implementations**: Extracted from service code

**Maintenance**: Run `python scripts/generate_docs.py` after code changes.

### 2. Manual Documentation

These documents require manual maintenance:

- **Architecture Diagrams**: C4 Model diagrams and ADRs
- **Algorithm Specifications**: Mathematical formulations and theory
- **Workflow Diagrams**: BPMN business and technical processes
- **Interface Contracts**: UML sequence and class diagrams

**Maintenance**: Update when architecture or processes change.

### 3. Hybrid Documentation

These combine auto-generated and manual content:

- **Deployment Guides**: Base structure auto-generated, details manual
- **User Guides**: Screenshots and examples manually maintained
- **Integration Guides**: Code examples auto-extracted, explanations manual

## Contributing Workflow

### For Code Changes

1. **Update Code Documentation**:
   ```python
   def normalize_score(raw_score: float, age_group_model_id: int, db: Session) -> float:
       """
       Normalize anomaly score using percentile ranking within age group.
       
       This algorithm transforms raw reconstruction loss values into interpretable 
       scores on a 0-100 scale where 0=no anomaly, 100=maximal anomaly.
       
       Args:
           raw_score: Raw reconstruction loss value
           age_group_model_id: Age group for comparison  
           db: Database session for historical data
           
       Returns:
           Normalized score on 0-100 scale
           
       Raises:
           ScoreNormalizationError: If normalization fails
           
       Example:
           >>> normalize_score(0.5, 8, db_session)
           75.2
       """
   ```

2. **Regenerate Documentation**:
   ```bash
   python scripts/generate_docs.py
   ```

3. **Review Generated Changes**:
   ```bash
   git diff docs/
   ```

4. **Update Manual Documentation** (if needed):
   - Architecture changes → Update C4 diagrams
   - New algorithms → Update algorithm specifications
   - Process changes → Update BPMN workflows

### For Architecture Changes

1. **Update Architecture Decision Records**:
   ```markdown
   # ADR-005: New Caching Strategy
   
   ## Status
   Accepted
   
   ## Context
   Current system experiences performance issues...
   
   ## Decision
   Implement Redis-based caching...
   
   ## Consequences
   - Improved response times
   - Additional infrastructure dependency
   ```

2. **Update C4 Diagrams**:
   - System Context: External dependencies
   - Container: Technology choices
   - Component: Internal structure
   - Code: Class relationships

3. **Update Interface Specifications**:
   - Service contracts
   - API specifications
   - Data models

### For Algorithm Changes

1. **Update Mathematical Specifications**:
   ```markdown
   ## Mathematical Formulation
   
   Given a set of historical scores S = {s₁, s₂, ..., sₙ}:
   
   Percentile Rank (PR) = (Number of scores ≤ x) / n × 100
   ```

2. **Update Implementation Documentation**:
   - Algorithm complexity analysis
   - Performance characteristics
   - Validation methods

3. **Update Property-Based Tests**:
   ```python
   @given(st.floats(min_value=0.0, max_value=10.0))
   def test_normalized_score_bounds(raw_score):
       """Property: Normalized scores must be in [0, 100] range."""
       normalized = normalize_score(raw_score, age_group_id, db)
       assert 0.0 <= normalized <= 100.0
   ```

## Documentation Standards

### Writing Style

- **Clear and Concise**: Use simple, direct language
- **Technical Accuracy**: Ensure all technical details are correct
- **Consistent Terminology**: Use the same terms throughout
- **Active Voice**: Prefer active over passive voice
- **Present Tense**: Document current system state

### Formatting Standards

#### Markdown Conventions
```markdown
# Main Title (H1)
## Section Title (H2)  
### Subsection Title (H3)

**Bold** for emphasis
*Italic* for technical terms
`code` for inline code
```

#### Code Examples
```python
# Always include complete, runnable examples
from app.services.score_normalizer import get_score_normalizer

normalizer = get_score_normalizer()
result = normalizer.normalize_score(0.5, 8, db)
print(f"Normalized score: {result}")
```

#### Diagrams
```mermaid
# Use Mermaid for diagrams when possible
sequenceDiagram
    participant User
    participant API
    participant Service
    
    User->>API: Upload drawing
    API->>Service: Process image
    Service-->>API: Analysis result
    API-->>User: Display results
```

### Documentation Review Process

1. **Technical Review**: Verify technical accuracy
2. **Editorial Review**: Check grammar and style
3. **User Testing**: Validate with target audience
4. **Accessibility Review**: Ensure inclusive design

## Tools and Automation

### Documentation Generation
```bash
# Generate all documentation
python scripts/generate_docs.py

# Generate specific categories
python scripts/generate_docs.py --api-only
python scripts/generate_docs.py --algorithms-only
```

### Validation Tools
```bash
# Check markdown formatting
markdownlint docs/

# Validate links
markdown-link-check docs/**/*.md

# Check spelling
cspell "docs/**/*.md"
```

### Diagram Tools

- **Architecture Diagrams**: Draw.io, Lucidchart
- **Sequence Diagrams**: Mermaid, PlantUML
- **API Documentation**: Swagger UI, ReDoc
- **Database Diagrams**: dbdiagram.io, ERDPlus

## Quality Checklist

### For New Documentation

- [ ] Follows established structure and naming conventions
- [ ] Uses appropriate industry standard format
- [ ] Includes complete and accurate technical details
- [ ] Contains working code examples
- [ ] Has been reviewed for technical accuracy
- [ ] Links are valid and functional
- [ ] Diagrams are clear and properly formatted
- [ ] Follows accessibility guidelines

### For Documentation Updates

- [ ] Changes reflect actual system behavior
- [ ] Version information is updated
- [ ] Related documentation is also updated
- [ ] Auto-generated content is refreshed
- [ ] Cross-references are maintained
- [ ] Deprecated information is removed

## Maintenance Schedule

### Daily
- Auto-generate documentation after code changes
- Validate links and formatting

### Weekly  
- Review and update manual documentation
- Check for outdated information
- Update screenshots and examples

### Monthly
- Comprehensive documentation review
- Update architecture diagrams
- Review and update ADRs
- Validate all external links

### Quarterly
- Major documentation restructuring if needed
- Update documentation standards
- Review and improve automation tools
- Conduct user feedback sessions

## Getting Help

### Documentation Issues
- **Technical Questions**: Ask in development team chat
- **Writing Help**: Consult technical writing guidelines
- **Tool Issues**: Check tool documentation or ask DevOps team

### Resources
- [C4 Model Documentation](https://c4model.com/)
- [OpenAPI Specification](https://swagger.io/specification/)
- [BPMN 2.0 Guide](https://www.bpmn.org/)
- [UML 2.5 Reference](https://www.uml.org/)
- [Markdown Guide](https://www.markdownguide.org/)

## Contact

For questions about documentation:
- **Technical Content**: Development Team
- **Process Documentation**: Product Team  
- **User Documentation**: UX Team
- **Infrastructure Documentation**: DevOps Team