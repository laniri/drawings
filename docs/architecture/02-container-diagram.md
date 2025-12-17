# Container Diagram (C4 Level 2)

## Overview

This diagram shows the high-level technology choices and how responsibilities are distributed across containers.

## Container Architecture

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryColor':'#3742fa', 'primaryTextColor':'#fff', 'primaryBorderColor':'#2f3542', 'lineColor':'#57606f', 'secondaryColor':'#2ed573', 'tertiaryColor':'#ffa502'}}}%%
C4Container
    title Container Diagram - System Architecture
    
    Person(user, "System User", "Interacts with the system")
    
    Container(container_0, "Backend API", "FastAPI/Python", "REST API server providing ML analysis and data management")
    Container(container_1, "Web Application", "React/JavaScript", "Single-page application providing user interface")
    Container(container_2, "Database", "SQLite/SQLAlchemy", "Relational database for metadata and results storage")
    Container(container_3, "File Storage", "Local Filesystem", "File storage for drawings and generated content")
    Container(container_4, "ML Model Store", "PyTorch/Pickle", "Storage and management of trained ML models")

    Rel(container_1, container_0, "Makes API calls", "HTTPS/REST")
    Rel(container_0, container_2, "Reads/Writes", "SQL")
```


## Legend

- **Person (Blue)**: System users
- **Container (Green)**: Deployable/executable units
- **Database (Yellow)**: Data storage containers
- **Relationship (Arrow)**: Communication between containers
## Container Details

### Backend API

**Technology**: FastAPI/Python

**Description**: REST API server providing ML analysis and data management

**Key Responsibilities**:
- Drawing upload and processing
- ML model inference
- Anomaly detection scoring
- Data persistence

### Web Application

**Technology**: React/JavaScript

**Description**: Single-page application providing user interface

**Key Responsibilities**:
- Drawing upload interface
- Analysis results visualization
- Configuration management
- Dashboard and reporting

### Database

**Technology**: SQLite/SQLAlchemy

**Description**: Relational database for metadata and results storage

**Key Responsibilities**:
- Drawing metadata storage
- Analysis results persistence
- Configuration data
- User session management

### File Storage

**Technology**: Local Filesystem

**Description**: File storage for drawings and generated content

**Key Responsibilities**:
- Drawing file storage
- ML model storage
- Generated visualizations
- Backup and export files

### ML Model Store

**Technology**: PyTorch/Pickle

**Description**: Storage and management of trained ML models

**Key Responsibilities**:
- Autoencoder model storage
- ViT feature extractor
- Model versioning
- Training artifacts

