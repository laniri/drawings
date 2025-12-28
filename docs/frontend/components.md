# Frontend Components

This document describes the React components in the application.

## Page Components

### AnalysisPage

**File**: `src/pages/AnalysisPage.tsx`

Main analysis results page with 6 interactive tabs for comprehensive drawing analysis.

### UploadPage

**File**: `src/pages/UploadPage.tsx`

Single drawing upload interface with subject category selection and metadata input.

**Props**: See TypeScript interface in source file

### BatchProcessingPage

**File**: `src/pages/BatchProcessingPage.tsx`

Batch upload interface for processing multiple drawings simultaneously.

**Props**: See TypeScript interface in source file

### ConfigurationPage

**File**: `src/pages/ConfigurationPage.tsx`

System configuration interface for threshold management and model settings.

### DashboardPage

**File**: `src/pages/DashboardPage.tsx`

Main dashboard with real-time statistics, age distribution, and system status.

### DocumentationPage

**File**: `src/pages/DocumentationPage.tsx`

Documentation management interface for generating and validating system documentation.

## Interpretability Components

The interpretability system provides comprehensive visual explanations for all drawing analyses with subject-aware context.

### ConfidenceIndicator

**File**: `src/components/interpretability/ConfidenceIndicator.tsx`

**Purpose**: Displays confidence metrics and reliability assessment for analysis results.

**Key Features**:
- Overall confidence scoring with visual indicators
- Model certainty, explanation reliability, and data sufficiency metrics
- Technical details breakdown with base model confidence and training data quality
- Subject-aware confidence warnings and recommendations
- Compact and full display modes
- Color-coded confidence levels (High: 80%+, Medium: 60-79%, Low: <60%)

**Props**:
- `analysisId: number` - ID of the analysis to show confidence for
- `showTechnicalDetails?: boolean` - Whether to show technical breakdown
- `compact?: boolean` - Whether to use compact display mode

### ExplanationLevelToggle

**File**: `src/components/interpretability/ExplanationLevelToggle.tsx`

**Purpose**: Toggles between technical and simplified explanations with user role adaptation.

**Key Features**:
- Technical vs simplified explanation modes
- User role-specific content (researcher, educator, parent, clinician)
- Subject-aware explanations with age-appropriate context
- Interactive accordions for key findings, visual indicators, and recommendations
- Confidence level explanations and visual cues guide

### ExportToolbar

**File**: `src/components/interpretability/ExportToolbar.tsx`

**Purpose**: Multi-format export functionality for analysis results.

**Key Features**:
- Export formats: PNG, PDF, JSON, CSV, HTML
- Subject-aware comprehensive reports
- Customizable export options and templates
- Batch export capabilities
- Progress tracking for export operations

### ExampleGallery

**File**: `src/components/interpretability/ExampleGallery.tsx`

**Purpose**: Gallery of example patterns for comparative analysis.

**Key Features**:
- Normal, anomalous, and borderline example patterns
- Age group and subject category filtering
- Interactive example selection with detailed views
- Pattern statistics and prevalence information
- Role-specific guidance for different user types

### ContextualHelpSystem

**File**: `src/components/interpretability/ContextualHelpSystem.tsx`

**Purpose**: Context-sensitive help system for interpretability features.

**Key Features**:
- Topic-specific help content (saliency maps, confidence scores, anomaly detection)
- User role-specific explanations
- Technical details toggle
- Interactive popover help with examples and tips
- Comprehensive coverage of interpretability concepts

### AdaptiveExplanationSystem

**File**: `src/components/interpretability/AdaptiveExplanationSystem.tsx`

**Purpose**: Adaptive explanation system that adjusts content based on user role and complexity preferences.

**Key Features**:
- Configurable complexity levels (1-5 scale)
- Auto-adaptation based on user role
- Multiple explanation styles (detailed, concise, visual)
- Vocabulary level adjustment (basic, intermediate, advanced)
- Dynamic content generation with subject context

## Shared Components

### SubjectCategorySelect

**File**: `src/components/SubjectCategorySelect.tsx`

**Purpose**: Subject category selection component with 64 predefined categories.

**Key Features**:
- Grouped categories (People & Body, Animals, Objects & Household, Transportation, Nature & Food, Abstract & Other)
- Search functionality with filtering
- Popular subjects quick selection
- Icon-based visual representation
- Form integration with React Hook Form

**Props**:
- `control: Control<any>` - React Hook Form control
- `name: string` - Form field name
- `label?: string` - Field label
- `required?: boolean` - Whether field is required
- `error?: boolean` - Error state
- `helperText?: string` - Help text
- `showSearch?: boolean` - Whether to show search functionality

## Layout Components

### Layout Components

**Directory**: `src/components/Layout/`

Contains layout-related components for consistent application structure and navigation.

