# Interpretability Implementation

## Current Implementation

- **Approach**: Simplified gradient-based saliency generation
- **Coverage**: Guaranteed interpretability for ALL drawings (not just anomalies)
- **Method**: Custom saliency map generation using PIL and image analysis
- **Storage**: Saliency maps stored in `static/saliency_maps/` directory

## API Endpoints

- `/api/v1/interpretability/{analysis_id}/interactive` - Interactive regions and patches
- `/api/v1/interpretability/{analysis_id}/confidence` - Confidence metrics
- `/api/v1/interpretability/{analysis_id}/export` - Multi-format exports
- `/api/v1/interpretability/{analysis_id}/annotate` - User annotations

## Export Formats

- **PNG**: Composite images with original + saliency side-by-side
- **PDF**: Comprehensive reports using ReportLab
- **JSON**: Complete structured data
- **CSV**: Tabular analysis data
- **HTML**: Web-ready reports

## Frontend Components

- `InteractiveInterpretabilityViewer` - Main interactive viewer
- `ExplanationLevelToggle` - Technical vs simplified explanations
- `ConfidenceIndicator` - Visual confidence metrics
- `ExportToolbar` - Multi-format export functionality
- `AnnotationTools` - User annotation system
- `ComparativeAnalysisPanel` - Comparison features
- `HistoricalInterpretationTracker` - Analysis history

## Dependencies

- **ReportLab**: Required for PDF generation (`pip install reportlab`)
- **Pillow**: Image processing and saliency map creation
- **OpenCV**: Advanced image analysis (optional)