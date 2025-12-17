# System Context Diagram (C4 Level 1)

## Overview

ML-powered system for analyzing children's drawings to detect developmental anomalies

## System Context

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryColor':'#ff6b6b', 'primaryTextColor':'#fff', 'primaryBorderColor':'#ff4757', 'lineColor':'#5f27cd', 'secondaryColor':'#00d2d3', 'tertiaryColor':'#ff9ff3'}}}%%
C4Context
    title System Context - Children's Drawing Anomaly Detection System
    
    Person_Ext(researcher, "Researcher", "Studies child development patterns using drawing analysis")
    Person_Ext(educator, "Educational Professional", "Monitors developmental milestones in children") 
    Person_Ext(healthcare, "Healthcare Provider", "Screens for developmental concerns")
    
    System(main_system, "Children's Drawing Anomaly Detection System", "ML-powered system for analyzing children's drawings to detect developmental anomalies")
    
    System_Ext(ext_sys_0, "File Storage System", "Stores uploaded drawings and generated visualizations")
    System_Ext(ext_sys_1, "ML Model Repository", "Stores trained autoencoder models and ViT embeddings")
    System_Ext(ext_sys_2, "AWS Services", "Optional cloud training and deployment services")
    System_Ext(ext_sys_3, "Container Registry", "Docker image storage and distribution")

    Rel(researcher, main_system, "Uploads drawings, views analysis", "HTTPS")
    Rel(educator, main_system, "Monitors progress, generates reports", "HTTPS")
    Rel(healthcare, main_system, "Screens patients, exports findings", "HTTPS")
    Rel(main_system, ext_sys_0, "Stores/retrieves files", "File I/O")
    Rel(main_system, ext_sys_1, "Loads models, saves training results", "File I/O")
    Rel(main_system, ext_sys_2, "Optional model training", "HTTPS/API")
    Rel(main_system, ext_sys_3, "Pulls/pushes images", "HTTPS")
```


## Legend

- **Person (Blue)**: External users of the system
- **System (Red)**: The main system being documented  
- **External System (Gray)**: External dependencies
- **Relationship (Arrow)**: Interaction between elements


## Diagram Validation


## System Users

### Researcher
Primary user type interacting with the Children's Drawing Anomaly Detection System.

### Educational Professional
Primary user type interacting with the Children's Drawing Anomaly Detection System.

### Healthcare Provider
Primary user type interacting with the Children's Drawing Anomaly Detection System.

## External Systems

### File Storage System
External dependency providing services to the main system.

### ML Model Repository
External dependency providing services to the main system.

### AWS Services
External dependency providing services to the main system.

### Container Registry
External dependency providing services to the main system.

