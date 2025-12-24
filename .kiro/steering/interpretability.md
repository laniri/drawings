# Interpretability Implementation

## Current Implementation

- **Approach**: Subject-aware simplified gradient-based saliency generation
- **Coverage**: Guaranteed interpretability for ALL drawings (not just anomalies) with subject context
- **Method**: Custom saliency map generation using PIL and image analysis with subject-specific comparisons
- **Storage**: Saliency maps stored in `static/saliency_maps/` directory with subject metadata

## API Endpoints

- `/api/v1/interpretability/{analysis_id}/interactive` - Interactive regions and patches with subject context
- `/api/v1/interpretability/{analysis_id}/confidence` - Subject-aware confidence metrics
- `/api/v1/interpretability/{analysis_id}/export` - Multi-format exports with subject information
- `/api/v1/interpretability/{analysis_id}/annotate` - User annotations with subject context

## Export Formats

- **PNG**: Composite images with original + saliency side-by-side including subject information
- **PDF**: Comprehensive reports using ReportLab with subject-aware analysis
- **JSON**: Complete structured data including subject metadata and hybrid embedding components
- **CSV**: Tabular analysis data with subject categories and confidence metrics
- **HTML**: Web-ready reports with subject-contextualized interpretability

## Frontend Components

- `InteractiveInterpretabilityViewer` - Main interactive viewer with subject-aware features
- `ExplanationLevelToggle` - Technical vs simplified explanations with subject context
- `ConfidenceIndicator` - Visual confidence metrics including subject-specific reliability
- `ExportToolbar` - Multi-format export functionality with subject information options
- `AnnotationTools` - User annotation system with subject context
- `ComparativeAnalysisPanel` - Subject-aware comparison features
- `HistoricalInterpretationTracker` - Analysis history with subject categorization

## Dependencies

- **ReportLab**: Required for PDF generation (`pip install reportlab`)
- **Pillow**: Core image processing and saliency map creation
- **OpenCV**: Advanced image analysis (optional, with PIL fallback)
- **Boto3**: AWS services for production deployment (optional for local development)