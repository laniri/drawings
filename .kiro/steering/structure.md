# Project Structure & Organization

## Root Directory Layout

```
├── app/                          # Python backend application
├── frontend/                     # React TypeScript frontend
├── alembic/                      # Database migrations
├── tests/                        # Backend test suite
├── uploads/                      # User-uploaded drawings
├── static/                       # Generated static files (models, saliency maps, exports)
├── .kiro/                        # Kiro IDE configuration
├── venv/                         # Python virtual environment
└── docker-compose.yml            # Container orchestration
```

## Backend Structure (`app/`)

```
app/
├── __init__.py
├── main.py                       # FastAPI application entry point
├── api/                          # API layer
│   └── api_v1/                   # Version 1 API endpoints
│       ├── api.py                # Router configuration
│       └── endpoints/            # Individual endpoint modules
│           ├── analysis.py       # Analysis operations
│           ├── config.py         # Configuration management
│           ├── database.py       # Database management operations
│           ├── drawings.py       # Drawing upload/retrieval
│           ├── interpretability.py # Advanced interpretability features
│           └── models.py         # ML model operations
├── core/                         # Core application configuration
│   ├── config.py                 # Settings and environment variables
│   ├── database.py               # Database connection setup
│   ├── exceptions.py             # Custom exception classes
│   └── middleware.py             # FastAPI middleware configuration
├── models/                       # Database models (SQLAlchemy)
│   └── database.py               # Table definitions
├── schemas/                      # Pydantic schemas for validation
│   ├── analysis.py               # Analysis request/response schemas
│   ├── common.py                 # Shared schema components
│   ├── drawings.py               # Drawing-related schemas
│   └── models.py                 # ML model schemas
├── services/                     # Business logic layer
│   ├── age_group_manager.py      # Age group classification
│   ├── backup_service.py         # Data backup and recovery
│   ├── comparison_service.py     # Model comparison utilities
│   ├── data_pipeline.py          # Data processing pipeline
│   ├── data_sufficiency_service.py # Training data validation
│   ├── dataset_preparation.py    # Dataset preparation for training
│   ├── embedding_service.py      # Hybrid feature extraction (ViT + subject encoding)
│   ├── file_storage.py           # File management
│   ├── health_monitor.py         # System health monitoring
│   ├── interpretability_engine.py # Subject-aware saliency map generation
│   ├── local_training_environment.py # Local model training
│   ├── model_deployment_service.py # Model deployment management
│   ├── model_manager.py          # Subject-aware ML model lifecycle
│   ├── sagemaker_training_service.py # AWS SageMaker integration
│   ├── score_normalizer.py       # Anomaly score normalization
│   ├── threshold_manager.py      # Subject-contextualized threshold configuration
│   ├── training_config.py        # Training configuration management
│   └── training_report_service.py # Training progress reporting
└── utils/                        # Utility functions
    └── embedding_serialization.py # Embedding data serialization
```

## Frontend Structure (`frontend/src/`)

```
src/
├── App.tsx                       # Main application component
├── main.tsx                      # Application entry point
├── components/                   # Reusable UI components
│   ├── Layout/                   # Layout components
│   └── interpretability/         # Advanced interpretability components
├── pages/                        # Page-level components
│   ├── AnalysisPage.tsx          # Analysis results view
│   ├── BatchProcessingPage.tsx   # Batch upload interface
│   ├── ConfigurationPage.tsx     # System configuration
│   ├── DashboardPage.tsx         # Main dashboard
│   └── UploadPage.tsx            # Single drawing upload
├── store/                        # State management
│   └── useAppStore.ts            # Zustand store configuration
├── test/                         # Frontend test suite
│   ├── AnalysisPage.test.tsx     # Analysis page tests
│   ├── ConfigurationPage.test.tsx # Configuration page tests
│   ├── UploadPage.test.tsx       # Upload page tests
│   ├── setup.ts                  # Test setup configuration
│   └── utils.tsx                 # Test utilities
└── theme/                        # Material-UI theming
    └── theme.ts                  # Theme configuration
```

## Key Architectural Patterns

### Backend Patterns
- **Layered Architecture**: API → Services → Models
- **Dependency Injection**: Settings via Pydantic
- **Repository Pattern**: Database access through SQLAlchemy
- **Service Layer**: Business logic isolated in `services/`
- **Schema Validation**: Pydantic for request/response validation

### Frontend Patterns
- **Component-Based**: React functional components with hooks
- **State Management**: Zustand for client state, React Query for server state
- **Form Handling**: React Hook Form with Zod validation
- **Routing**: React Router for navigation
- **Styling**: Material-UI with custom theme

## File Naming Conventions

### Backend
- **Modules**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`

### Frontend
- **Components**: `PascalCase.tsx`
- **Hooks**: `use[Name].ts`
- **Utilities**: `camelCase.ts`
- **Types**: `PascalCase` interfaces/types

## Import Organization

### Backend (isort configuration)
1. Standard library imports
2. Third-party imports
3. Local application imports (`from app.`)

### Frontend
1. React and React-related imports
2. Third-party library imports
3. Local component imports (using `@/` alias)
4. Type-only imports (with `type` keyword)