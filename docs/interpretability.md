# Interpretability System Documentation

## Overview

The Children's Drawing Anomaly Detection System includes a comprehensive interpretability engine that provides visual explanations and confidence assessments for all drawing analyses. The system uses subject-aware saliency generation with guaranteed coverage for every drawing, ensuring users can understand and trust the AI's decisions.

## Core Features

### Subject-Aware Saliency Generation
- **Method**: Simplified gradient-based saliency mapping with subject context
- **Coverage**: Guaranteed interpretability for ALL drawings (not just anomalies)
- **Storage**: Saliency maps stored in `static/saliency_maps/` with subject metadata
- **Fallback**: PIL-based processing when OpenCV is unavailable

### Multi-Dimensional Confidence Assessment
- **Overall Confidence**: Composite score combining multiple reliability factors
- **Model Certainty**: How confident the AI model is about its prediction
- **Explanation Reliability**: Trustworthiness of visual explanations and saliency maps
- **Data Sufficiency**: Quality and quantity of training data for the age group
- **Technical Breakdown**: Detailed metrics for researchers and technical users

### Adaptive Explanation System
- **Role-Based Content**: Customized explanations for researchers, educators, parents, and clinicians
- **Complexity Levels**: 5-level complexity scale from very simple to expert
- **Vocabulary Adaptation**: Basic, intermediate, and advanced terminology
- **Auto-Adaptation**: Automatic adjustment based on user role preferences

## Frontend Components

### ConfidenceIndicator
**Purpose**: Displays comprehensive confidence metrics and reliability assessment

**Key Features**:
- Visual confidence meters with color-coded levels
- Technical details breakdown for advanced users
- Subject-aware warnings and recommendations
- Compact mode for space-constrained displays
- Interpretation guidance for different confidence levels

**Usage**:
```tsx
<ConfidenceIndicator 
  analysisId={522} 
  showTechnicalDetails={true}
  compact={false}
/>
```

### ExplanationLevelToggle
**Purpose**: Switches between technical and simplified explanations

**Key Features**:
- Technical vs simplified explanation modes
- User role-specific content adaptation
- Subject-aware explanations with age-appropriate context
- Interactive accordions for organized information display
- Visual cues guide for saliency map interpretation

**Usage**:
```tsx
<ExplanationLevelToggle
  analysisId={522}
  technicalExplanation="Detailed model output..."
  userRole="educator"
  onExplanationChange={(level, explanation) => console.log(level, explanation)}
/>
```

### ExampleGallery
**Purpose**: Provides comparative examples for pattern understanding

**Key Features**:
- Normal, anomalous, and borderline pattern examples
- Age group and subject category filtering
- Pattern statistics and prevalence information
- Interactive example selection with detailed views
- Role-specific guidance for interpretation

**Usage**:
```tsx
<ExampleGallery
  ageGroup="5-6"
  userRole="educator"
  filterByType="all"
  onExampleSelect={(example) => console.log(example)}
/>
```

### ContextualHelpSystem
**Purpose**: Context-sensitive help for interpretability features

**Key Features**:
- Topic-specific help content (saliency maps, confidence scores, etc.)
- User role-specific explanations
- Technical details toggle
- Interactive popover interface
- Comprehensive coverage of interpretability concepts

**Usage**:
```tsx
<ContextualHelpSystem
  topic="saliency-maps"
  userRole="parent"
  placement="top"
  showTechnical={false}
/>
```

### AdaptiveExplanationSystem
**Purpose**: Generates adaptive explanations based on user preferences

**Key Features**:
- Configurable complexity levels (1-5 scale)
- Multiple explanation styles (detailed, concise, visual)
- Vocabulary level adjustment
- Auto-adaptation based on user role
- Dynamic content generation with subject context

**Usage**:
```tsx
<AdaptiveExplanationSystem
  analysisData={analysisResult}
  onConfigChange={(config) => console.log(config)}
  initialConfig={{ userRole: 'educator', complexity: 3 }}
/>
```

### ExportToolbar
**Purpose**: Multi-format export functionality for analysis results

**Key Features**:
- Export formats: PNG, PDF, JSON, CSV, HTML
- Subject-aware comprehensive reports
- Customizable export options
- Batch export capabilities
- Progress tracking for export operations

## API Endpoints

### Interactive Interpretability
```
GET /api/v1/interpretability/{analysis_id}/interactive
```
Returns interactive regions and patches with subject context for hoverable saliency maps.

### Confidence Metrics
```
GET /api/v1/interpretability/{analysis_id}/confidence
```
Returns comprehensive confidence assessment including:
- Overall confidence score
- Model certainty metrics
- Explanation reliability
- Data sufficiency assessment
- Technical breakdown for advanced users

### Export Analysis
```
POST /api/v1/interpretability/{analysis_id}/export
```
Exports analysis results in multiple formats with subject-aware comprehensive reports.

**Request Body**:
```json
{
  "format": "pdf",
  "export_options": {
    "include_subject_context": true,
    "include_technical_details": false,
    "user_role": "educator"
  }
}
```

### User Annotations
```
POST /api/v1/interpretability/{analysis_id}/annotate
```
Allows users to add annotations to analysis results with subject context.

## Export Formats

### PNG Export
- Composite images with original + saliency side-by-side
- Subject information overlay
- Confidence indicators
- High-resolution output suitable for presentations

### PDF Export (requires ReportLab)
- Comprehensive multi-page reports
- Subject-aware analysis summary
- Embedded charts and visualizations
- Professional formatting for clinical/research use

### JSON Export
- Complete structured data
- Subject metadata and hybrid embedding components
- Confidence metrics and technical details
- Machine-readable format for further analysis

### CSV Export
- Tabular analysis data
- Subject categories and confidence metrics
- Suitable for statistical analysis
- Compatible with spreadsheet applications

### HTML Export
- Web-ready interactive reports
- Subject-contextualized interpretability
- Embedded visualizations
- Shareable format for web distribution

## Configuration and Dependencies

### Required Dependencies
- **Pillow**: Core image processing and saliency map creation
- **NumPy**: Numerical computations for saliency generation
- **Matplotlib**: Colormap generation for saliency visualization

### Optional Dependencies
- **OpenCV**: Advanced image processing (falls back to PIL if unavailable)
- **ReportLab**: PDF generation for comprehensive reports
- **Boto3**: AWS services for production deployment

### Configuration Options
The interpretability system can be configured through environment variables:

```bash
# Enable/disable interpretability features
INTERPRETABILITY_ENABLED=true

# Saliency map storage directory
SALIENCY_MAPS_DIR=static/saliency_maps

# Default user role for explanations
DEFAULT_USER_ROLE=educator

# Enable technical details by default
SHOW_TECHNICAL_DETAILS=false
```

## User Roles and Adaptations

### Researcher
- **Complexity**: High (level 4-5)
- **Vocabulary**: Advanced technical terminology
- **Content**: Statistical significance, model architecture details, research applications
- **Explanations**: Detailed technical breakdowns with quantitative metrics

### Educator
- **Complexity**: Moderate (level 3)
- **Vocabulary**: Intermediate educational terminology
- **Content**: Developmental context, classroom applications, milestone tracking
- **Explanations**: Educational guidance with practical applications

### Parent
- **Complexity**: Simple (level 2)
- **Vocabulary**: Basic, accessible language
- **Content**: What results mean for child development, when to seek help
- **Explanations**: Clear, reassuring explanations with actionable guidance

### Clinician
- **Complexity**: High (level 4)
- **Vocabulary**: Clinical and medical terminology
- **Content**: Diagnostic implications, assessment recommendations, clinical context
- **Explanations**: Professional assessment support with clinical considerations

## Confidence Level Interpretation

### High Confidence (80%+)
- Strong evidence supports the analysis
- Results are likely reliable for decision-making
- Model has high certainty based on training data
- Explanations are trustworthy and actionable

### Medium Confidence (60-79%)
- Moderate evidence supports the analysis
- Consider additional context or assessment
- Model shows reasonable certainty
- Use results as one factor among others

### Low Confidence (<60%)
- Limited evidence available
- Use results cautiously
- Seek additional professional input
- Consider as preliminary screening only

## Technical Implementation

### Saliency Map Generation
The system uses a simplified gradient-based approach optimized for reliability:

1. **Feature Extraction**: Vision Transformer processes the drawing
2. **Gradient Computation**: Calculates gradients with respect to reconstruction loss
3. **Saliency Mapping**: Generates attention-based saliency maps
4. **Subject Context**: Incorporates subject category information
5. **Visualization**: Creates color-coded overlays with PIL/OpenCV

### Confidence Calculation
Multi-factor confidence assessment:

```python
overall_confidence = weighted_average([
    base_model_confidence * 0.4,
    training_data_quality * 0.3,
    explanation_reliability * 0.2,
    score_extremity * 0.1
])
```

### Storage and Caching
- Saliency maps cached in `static/saliency_maps/`
- Confidence metrics cached with analysis results
- Subject metadata stored with interpretability data
- Automatic cleanup of old cached files

## Best Practices

### For Developers
1. Always provide fallback explanations when technical details fail
2. Use appropriate user role adaptations
3. Include confidence indicators with all interpretability features
4. Test with different user roles and complexity levels
5. Ensure graceful degradation when optional dependencies are missing

### For Users
1. Consider confidence levels when interpreting results
2. Use role-appropriate explanation levels
3. Combine interpretability with domain expertise
4. Export results for documentation and sharing
5. Provide feedback through annotation tools

## Troubleshooting

### Common Issues

**Saliency maps not generating**:
- Check that PIL/OpenCV dependencies are installed
- Verify image preprocessing pipeline
- Ensure sufficient memory for gradient computation

**Low confidence scores**:
- May indicate insufficient training data for age group
- Consider subject category representation in training set
- Review model performance metrics

**Export failures**:
- Check ReportLab installation for PDF exports
- Verify file permissions for export directory
- Ensure sufficient disk space for large exports

**Performance issues**:
- Saliency generation is computationally intensive
- Consider caching strategies for frequently accessed analyses
- Monitor memory usage during batch processing

## Future Enhancements

- **Real-time Saliency**: Live saliency updates during drawing upload
- **Interactive Annotations**: Enhanced annotation tools with collaborative features
- **Advanced Visualizations**: 3D saliency maps and temporal analysis
- **Custom Explanations**: User-defined explanation templates
- **Multi-language Support**: Explanations in multiple languages