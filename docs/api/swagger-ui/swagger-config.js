// Swagger UI Configuration

function initializeSwaggerUI() {
    // Store the original spec
    const spec = {
  "openapi": "3.1.0",
  "info": {
    "title": "Children's Drawing Anomaly Detection System",
    "description": "Machine learning system for detecting anomalies in children's drawings",
    "version": "0.1.0"
  },
  "paths": {
    "/health": {
      "get": {
        "summary": "Health Check",
        "description": "Lightweight health check endpoint for load balancer.",
        "operationId": "health_check_health_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/health/simple": {
      "get": {
        "summary": "Simple Health Check",
        "description": "Ultra-lightweight health check for ALB - no dependencies.",
        "operationId": "simple_health_check_health_simple_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/v1/drawings/upload": {
      "post": {
        "tags": [
          "drawings"
        ],
        "summary": "Upload Drawing",
        "description": "Upload drawing with metadata.\n\nThis endpoint accepts multipart form data with an image file and metadata.\nThe image is validated, preprocessed, and stored along with the metadata.",
        "operationId": "upload_drawing_api_v1_drawings_upload_post",
        "requestBody": {
          "content": {
            "multipart/form-data": {
              "schema": {
                "$ref": "#/components/schemas/Body_upload_drawing_api_v1_drawings_upload_post"
              }
            }
          },
          "required": true
        },
        "responses": {
          "201": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/DrawingResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/drawings/upload/progress/{upload_id}": {
      "get": {
        "tags": [
          "drawings"
        ],
        "summary": "Get Upload Progress",
        "description": "Get upload progress for large file uploads.",
        "operationId": "get_upload_progress_api_v1_drawings_upload_progress__upload_id__get",
        "parameters": [
          {
            "name": "upload_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Upload Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/drawings/{drawing_id}": {
      "get": {
        "tags": [
          "drawings"
        ],
        "summary": "Get Drawing",
        "description": "Retrieve drawing details by ID.",
        "operationId": "get_drawing_api_v1_drawings__drawing_id__get",
        "parameters": [
          {
            "name": "drawing_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Drawing Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/DrawingResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      },
      "delete": {
        "tags": [
          "drawings"
        ],
        "summary": "Delete Drawing",
        "description": "Delete drawing and associated data.",
        "operationId": "delete_drawing_api_v1_drawings__drawing_id__delete",
        "parameters": [
          {
            "name": "drawing_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Drawing Id"
            }
          }
        ],
        "responses": {
          "204": {
            "description": "Successful Response"
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/drawings/{drawing_id}/file": {
      "get": {
        "tags": [
          "drawings"
        ],
        "summary": "Get Drawing File",
        "description": "Retrieve the actual drawing file.",
        "operationId": "get_drawing_file_api_v1_drawings__drawing_id__file_get",
        "parameters": [
          {
            "name": "drawing_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Drawing Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/drawings/": {
      "get": {
        "tags": [
          "drawings"
        ],
        "summary": "List Drawings",
        "description": "List drawings with optional filtering and pagination.",
        "operationId": "list_drawings_api_v1_drawings__get",
        "parameters": [
          {
            "name": "age_min",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "number"
                },
                {
                  "type": "null"
                }
              ],
              "title": "Age Min"
            }
          },
          {
            "name": "age_max",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "number"
                },
                {
                  "type": "null"
                }
              ],
              "title": "Age Max"
            }
          },
          {
            "name": "subject",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ],
              "title": "Subject"
            }
          },
          {
            "name": "expert_label",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "$ref": "#/components/schemas/ExpertLabel"
                },
                {
                  "type": "null"
                }
              ],
              "title": "Expert Label"
            }
          },
          {
            "name": "page",
            "in": "query",
            "required": false,
            "schema": {
              "type": "integer",
              "default": 1,
              "title": "Page"
            }
          },
          {
            "name": "page_size",
            "in": "query",
            "required": false,
            "schema": {
              "type": "integer",
              "default": 20,
              "title": "Page Size"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/DrawingListResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/drawings/batch/upload": {
      "post": {
        "tags": [
          "drawings"
        ],
        "summary": "Batch Upload Drawings",
        "description": "Upload multiple drawings in batch.\n\nThis endpoint accepts multiple files and processes them in the background.\nReturns an upload ID for tracking progress.",
        "operationId": "batch_upload_drawings_api_v1_drawings_batch_upload_post",
        "requestBody": {
          "content": {
            "multipart/form-data": {
              "schema": {
                "$ref": "#/components/schemas/Body_batch_upload_drawings_api_v1_drawings_batch_upload_post"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/drawings/stats": {
      "get": {
        "tags": [
          "drawings"
        ],
        "summary": "Get Drawing Stats",
        "description": "Get statistics about stored drawings.",
        "operationId": "get_drawing_stats_api_v1_drawings_stats_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/v1/analysis/stats": {
      "get": {
        "tags": [
          "analysis"
        ],
        "summary": "Get Analysis Stats",
        "description": "Get dashboard statistics for analyses and drawings.\n\nThis endpoint provides comprehensive statistics for the dashboard\nincluding drawing counts, analysis results, and model status.",
        "operationId": "get_analysis_stats_api_v1_analysis_stats_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/v1/analysis/analyze/{drawing_id}": {
      "post": {
        "tags": [
          "analysis"
        ],
        "summary": "Analyze Drawing",
        "description": "Analyze specific drawing for anomalies.\n\nThis endpoint performs anomaly detection on a single drawing,\ngenerating embeddings, computing anomaly scores, and providing\ninterpretability results if the drawing is flagged as anomalous.",
        "operationId": "analyze_drawing_api_v1_analysis_analyze__drawing_id__post",
        "parameters": [
          {
            "name": "drawing_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Drawing Id"
            }
          }
        ],
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/AnalysisRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/AnalysisResultResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/analysis/batch": {
      "post": {
        "tags": [
          "analysis"
        ],
        "summary": "Batch Analyze",
        "description": "Batch analyze multiple drawings.\n\nThis endpoint accepts a list of drawing IDs and processes them\nin the background, returning a batch ID for progress tracking.",
        "operationId": "batch_analyze_api_v1_analysis_batch_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/BatchAnalysisRequest"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "additionalProperties": true,
                  "type": "object",
                  "title": "Response Batch Analyze Api V1 Analysis Batch Post"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/analysis/batch/{batch_id}/progress": {
      "get": {
        "tags": [
          "analysis"
        ],
        "summary": "Get Batch Progress",
        "description": "Get progress of batch analysis.",
        "operationId": "get_batch_progress_api_v1_analysis_batch__batch_id__progress_get",
        "parameters": [
          {
            "name": "batch_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Batch Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/BatchAnalysisResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/analysis/{analysis_id}": {
      "get": {
        "tags": [
          "analysis"
        ],
        "summary": "Get Analysis Result",
        "description": "Get analysis results by analysis ID.\n\nThis endpoint retrieves a complete analysis result including\nthe drawing information, anomaly analysis, and interpretability\nresults if available.",
        "operationId": "get_analysis_result_api_v1_analysis__analysis_id__get",
        "parameters": [
          {
            "name": "analysis_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Analysis Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/AnalysisResultResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/analysis/embeddings/{drawing_id}": {
      "post": {
        "tags": [
          "analysis"
        ],
        "summary": "Generate Embedding",
        "description": "Generate embedding for a drawing without requiring a trained model.\n\nThis endpoint is used during the training phase to generate embeddings\nfor all drawings before training the autoencoder models.",
        "operationId": "generate_embedding_api_v1_analysis_embeddings__drawing_id__post",
        "parameters": [
          {
            "name": "drawing_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Drawing Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/analysis/drawing/{drawing_id}": {
      "get": {
        "tags": [
          "analysis"
        ],
        "summary": "Get Drawing Analyses",
        "description": "Get all analyses for a specific drawing.\n\nThis endpoint returns the analysis history for a drawing,\nordered by most recent first.",
        "operationId": "get_drawing_analyses_api_v1_analysis_drawing__drawing_id__get",
        "parameters": [
          {
            "name": "drawing_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Drawing Id"
            }
          },
          {
            "name": "limit",
            "in": "query",
            "required": false,
            "schema": {
              "type": "integer",
              "default": 10,
              "title": "Limit"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/AnalysisHistoryResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/interpretability/{analysis_id}/interactive": {
      "get": {
        "tags": [
          "interpretability"
        ],
        "summary": "Get Interactive Interpretability",
        "description": "Get interactive saliency data with hoverable regions and click explanations.\n\nThis endpoint provides enhanced interpretability data that supports\ninteractive user interfaces with hover explanations and click-to-zoom functionality.",
        "operationId": "get_interactive_interpretability_api_v1_interpretability__analysis_id__interactive_get",
        "parameters": [
          {
            "name": "analysis_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Analysis Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/InteractiveInterpretabilityResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/interpretability/{analysis_id}/simplified": {
      "get": {
        "tags": [
          "interpretability"
        ],
        "summary": "Get Simplified Explanation",
        "description": "Get simplified, non-technical explanations suitable for educators and parents.\n\nThis endpoint provides explanations adapted for different user roles\nwith accessible language and clear recommendations.",
        "operationId": "get_simplified_explanation_api_v1_interpretability__analysis_id__simplified_get",
        "parameters": [
          {
            "name": "analysis_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Analysis Id"
            }
          },
          {
            "name": "user_role",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ],
              "default": "educator",
              "title": "User Role"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SimplifiedExplanationResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/interpretability/{analysis_id}/confidence": {
      "get": {
        "tags": [
          "interpretability"
        ],
        "summary": "Get Confidence Metrics",
        "description": "Get confidence metrics and reliability scores for interpretability results.\n\nThis endpoint provides detailed confidence information to help users\nassess the trustworthiness of the analysis and interpretations.",
        "operationId": "get_confidence_metrics_api_v1_interpretability__analysis_id__confidence_get",
        "parameters": [
          {
            "name": "analysis_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Analysis Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ConfidenceMetricsResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/interpretability/{analysis_id}/export": {
      "post": {
        "tags": [
          "interpretability"
        ],
        "summary": "Export Interpretability Results",
        "description": "Export interpretability results in multiple formats (PDF, PNG, CSV, JSON, HTML).\n\nThis endpoint allows users to export comprehensive interpretability reports\nwith customizable options for different use cases.",
        "operationId": "export_interpretability_results_api_v1_interpretability__analysis_id__export_post",
        "parameters": [
          {
            "name": "analysis_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Analysis Id"
            }
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ExportRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ExportResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/interpretability/examples": {
      "get": {
        "tags": [
          "interpretability"
        ],
        "summary": "Get Example Patterns",
        "description": "Get example interpretation patterns for educational purposes.\n\nThis endpoint provides a gallery of common interpretation patterns\nwith explanations suitable for different user roles.",
        "operationId": "get_example_patterns_api_v1_interpretability_examples_get",
        "parameters": [
          {
            "name": "age_group",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ],
              "title": "Age Group"
            }
          },
          {
            "name": "user_role",
            "in": "query",
            "required": false,
            "schema": {
              "type": "string",
              "default": "educator",
              "title": "User Role"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "type": "array",
                  "items": {
                    "type": "object",
                    "additionalProperties": true
                  },
                  "title": "Response Get Example Patterns Api V1 Interpretability Examples Get"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/interpretability/examples/{age_group}": {
      "get": {
        "tags": [
          "interpretability"
        ],
        "summary": "Get Comparison Examples",
        "description": "Get comparison examples for educational purposes from a specific age group.\n\nThis endpoint provides examples of normal and anomalous drawings\nto help users understand typical patterns and variations. Now supports\nfiltering by subject category for more targeted comparisons.",
        "operationId": "get_comparison_examples_api_v1_interpretability_examples__age_group__get",
        "parameters": [
          {
            "name": "age_group",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Age Group"
            }
          },
          {
            "name": "example_type",
            "in": "query",
            "required": false,
            "schema": {
              "type": "string",
              "default": "both",
              "title": "Example Type"
            }
          },
          {
            "name": "subject",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ],
              "title": "Subject"
            }
          },
          {
            "name": "limit",
            "in": "query",
            "required": false,
            "schema": {
              "type": "integer",
              "default": 5,
              "title": "Limit"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ComparisonExamplesResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/interpretability/{analysis_id}/attribution": {
      "get": {
        "tags": [
          "interpretability"
        ],
        "summary": "Get Anomaly Attribution",
        "description": "Get detailed anomaly attribution breakdown (age vs subject vs visual).\n\nThis endpoint provides detailed information about what contributed\nto the anomaly detection: age-related factors, subject-specific factors,\nor visual characteristics.",
        "operationId": "get_anomaly_attribution_api_v1_interpretability__analysis_id__attribution_get",
        "parameters": [
          {
            "name": "analysis_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Analysis Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/interpretability/{analysis_id}/annotate": {
      "post": {
        "tags": [
          "interpretability"
        ],
        "summary": "Add Annotation",
        "description": "Add user annotations to interpretability results.\n\nThis endpoint allows users to add their own notes and observations\nto interpretability results for future reference.",
        "operationId": "add_annotation_api_v1_interpretability__analysis_id__annotate_post",
        "parameters": [
          {
            "name": "analysis_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Analysis Id"
            }
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/AnnotationRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/models/age-groups": {
      "get": {
        "tags": [
          "models"
        ],
        "summary": "List Age Group Models",
        "description": "List available age group models.\n\nThis endpoint returns all age group models with their status,\nsample counts, and threshold information.",
        "operationId": "list_age_group_models_api_v1_models_age_groups_get",
        "parameters": [
          {
            "name": "active_only",
            "in": "query",
            "required": false,
            "schema": {
              "type": "boolean",
              "default": true,
              "title": "Active Only"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ModelListResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/models/train": {
      "post": {
        "tags": [
          "models"
        ],
        "summary": "Train Age Group Model",
        "description": "Train new age group model.\n\nThis endpoint starts training a new autoencoder model for the specified\nage range. Training is performed in the background and progress can be\ntracked using the returned job ID.",
        "operationId": "train_age_group_model_api_v1_models_train_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ModelTrainingRequest"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/models/training/{job_id}/status": {
      "get": {
        "tags": [
          "models"
        ],
        "summary": "Get Training Status",
        "description": "Get training job status.",
        "operationId": "get_training_status_api_v1_models_training__job_id__status_get",
        "parameters": [
          {
            "name": "job_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Job Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/models/{model_id}/threshold": {
      "put": {
        "tags": [
          "models"
        ],
        "summary": "Update Model Threshold",
        "description": "Update model threshold.\n\nThis endpoint allows updating the anomaly detection threshold\nfor a specific age group model. The threshold can be set directly\nor calculated from a percentile of validation data.",
        "operationId": "update_model_threshold_api_v1_models__model_id__threshold_put",
        "parameters": [
          {
            "name": "model_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Model Id"
            }
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ThresholdUpdateRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "additionalProperties": true,
                  "title": "Response Update Model Threshold Api V1 Models  Model Id  Threshold Put"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/models/status": {
      "get": {
        "tags": [
          "models"
        ],
        "summary": "Get Model Status",
        "description": "Get model training and system status.\n\nThis endpoint provides an overview of the model management system,\nincluding counts of models in different states and overall system health.",
        "operationId": "get_model_status_api_v1_models_status_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ModelStatusResponse"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/models/auto-create": {
      "post": {
        "tags": [
          "models"
        ],
        "summary": "Auto Create Age Groups",
        "description": "Automatically create age group models based on data distribution.\n\nThis endpoint analyzes the available drawing data and creates\nappropriate age group models with sufficient sample sizes.",
        "operationId": "auto_create_age_groups_api_v1_models_auto_create_post",
        "parameters": [
          {
            "name": "force_recreate",
            "in": "query",
            "required": false,
            "schema": {
              "type": "boolean",
              "default": false,
              "title": "Force Recreate"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/models/creation/{job_id}/status": {
      "get": {
        "tags": [
          "models"
        ],
        "summary": "Get Creation Status",
        "description": "Get model creation job status.",
        "operationId": "get_creation_status_api_v1_models_creation__job_id__status_get",
        "parameters": [
          {
            "name": "job_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Job Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/models/{model_id}": {
      "delete": {
        "tags": [
          "models"
        ],
        "summary": "Delete Model",
        "description": "Delete (deactivate) an age group model.\n\nThis endpoint deactivates a model rather than permanently deleting it\nto preserve analysis history.",
        "operationId": "delete_model_api_v1_models__model_id__delete",
        "parameters": [
          {
            "name": "model_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Model Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/models/data-sufficiency/analyze": {
      "get": {
        "tags": [
          "models"
        ],
        "summary": "Analyze Data Sufficiency",
        "description": "Analyze data sufficiency for age groups.\n\nThis endpoint analyzes the available data for specified age groups\nand provides warnings about insufficient data, unbalanced distributions,\nand other data quality issues.\n\nArgs:\n    age_groups: Comma-separated list of age ranges (e.g., \"3-4,4-5,5-6\")\n               If not provided, analyzes all existing age group models",
        "operationId": "analyze_data_sufficiency_api_v1_models_data_sufficiency_analyze_get",
        "parameters": [
          {
            "name": "age_groups",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ],
              "title": "Age Groups"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/models/data-sufficiency/age-group/{age_min}/{age_max}": {
      "get": {
        "tags": [
          "models"
        ],
        "summary": "Analyze Specific Age Group",
        "description": "Analyze data sufficiency for a specific age group.\n\nThis endpoint provides detailed analysis of data availability,\nquality, and distribution for a single age group.",
        "operationId": "analyze_specific_age_group_api_v1_models_data_sufficiency_age_group__age_min___age_max__get",
        "parameters": [
          {
            "name": "age_min",
            "in": "path",
            "required": true,
            "schema": {
              "type": "number",
              "title": "Age Min"
            }
          },
          {
            "name": "age_max",
            "in": "path",
            "required": true,
            "schema": {
              "type": "number",
              "title": "Age Max"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/models/data-sufficiency/merge-age-groups": {
      "post": {
        "tags": [
          "models"
        ],
        "summary": "Merge Age Groups",
        "description": "Merge age groups to improve data sufficiency.\n\nThis endpoint deactivates the original age group models and creates\na new merged age group model with combined data.",
        "operationId": "merge_age_groups_api_v1_models_data_sufficiency_merge_age_groups_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/Body_merge_age_groups_api_v1_models_data_sufficiency_merge_age_groups_post"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/models/data-sufficiency/warnings": {
      "get": {
        "tags": [
          "models"
        ],
        "summary": "Get Data Warnings",
        "description": "Get data sufficiency warnings for all age groups.\n\nThis endpoint returns warnings about data quality issues,\noptionally filtered by severity level.",
        "operationId": "get_data_warnings_api_v1_models_data_sufficiency_warnings_get",
        "parameters": [
          {
            "name": "severity",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ],
              "title": "Severity"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/training/jobs": {
      "post": {
        "tags": [
          "training"
        ],
        "summary": "Submit Training Job",
        "description": "Submit a new training job to either local or SageMaker environment.\n\nThis endpoint creates and submits a training job based on the specified\nenvironment. For SageMaker jobs, it handles container building, data upload,\nand job submission. For local jobs, it starts training immediately.",
        "operationId": "submit_training_job_api_v1_training_jobs_post",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/TrainingConfigRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/TrainingJobResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      },
      "get": {
        "tags": [
          "training"
        ],
        "summary": "List Training Jobs",
        "description": "List training jobs with optional filtering.\n\nThis endpoint returns a list of training jobs, optionally filtered\nby environment (local/sagemaker) and status.",
        "operationId": "list_training_jobs_api_v1_training_jobs_get",
        "parameters": [
          {
            "name": "environment",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "$ref": "#/components/schemas/TrainingEnvironment"
                },
                {
                  "type": "null"
                }
              ],
              "title": "Environment"
            }
          },
          {
            "name": "status",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ],
              "title": "Status"
            }
          },
          {
            "name": "limit",
            "in": "query",
            "required": false,
            "schema": {
              "type": "integer",
              "default": 50,
              "title": "Limit"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "type": "array",
                  "items": {
                    "$ref": "#/components/schemas/TrainingJobResponse"
                  },
                  "title": "Response List Training Jobs Api V1 Training Jobs Get"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/training/jobs/{job_id}": {
      "get": {
        "tags": [
          "training"
        ],
        "summary": "Get Training Job Status",
        "description": "Get detailed status of a specific training job.\n\nThis endpoint returns comprehensive information about a training job,\nincluding progress, metrics, and environment-specific details.",
        "operationId": "get_training_job_status_api_v1_training_jobs__job_id__get",
        "parameters": [
          {
            "name": "job_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Job Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "additionalProperties": true,
                  "title": "Response Get Training Job Status Api V1 Training Jobs  Job Id  Get"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/training/jobs/{job_id}/cancel": {
      "post": {
        "tags": [
          "training"
        ],
        "summary": "Cancel Training Job",
        "description": "Cancel a running training job.\n\nThis endpoint attempts to cancel a training job. For local jobs,\nit stops the training process. For SageMaker jobs, it stops the\nSageMaker training job.",
        "operationId": "cancel_training_job_api_v1_training_jobs__job_id__cancel_post",
        "parameters": [
          {
            "name": "job_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Job Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/training/jobs/{job_id}/reports": {
      "get": {
        "tags": [
          "training"
        ],
        "summary": "Get Training Reports",
        "description": "Get training reports for a specific job.\n\nThis endpoint returns all training reports associated with a job,\nincluding metrics, model paths, and performance summaries.",
        "operationId": "get_training_reports_api_v1_training_jobs__job_id__reports_get",
        "parameters": [
          {
            "name": "job_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Job Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "type": "array",
                  "items": {
                    "$ref": "#/components/schemas/TrainingReportResponse"
                  },
                  "title": "Response Get Training Reports Api V1 Training Jobs  Job Id  Reports Get"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/training/deploy": {
      "post": {
        "tags": [
          "training"
        ],
        "summary": "Deploy Trained Model",
        "description": "Deploy trained model parameters to production system.\n\nThis endpoint loads trained model parameters and creates a new\nage group model for production use.",
        "operationId": "deploy_trained_model_api_v1_training_deploy_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ModelDeploymentRequest"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/training/environments/status": {
      "get": {
        "tags": [
          "training"
        ],
        "summary": "Get Training Environments Status",
        "description": "Get status of available training environments.\n\nThis endpoint returns information about local and SageMaker\ntraining environments, including availability and configuration.",
        "operationId": "get_training_environments_status_api_v1_training_environments_status_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/v1/training/sagemaker/setup": {
      "post": {
        "tags": [
          "training"
        ],
        "summary": "Setup Sagemaker Environment",
        "description": "Setup SageMaker training environment.\n\nThis endpoint helps set up the necessary AWS resources for\nSageMaker training, including IAM roles and container repositories.",
        "operationId": "setup_sagemaker_environment_api_v1_training_sagemaker_setup_post",
        "parameters": [
          {
            "name": "s3_bucket",
            "in": "query",
            "required": true,
            "schema": {
              "type": "string",
              "title": "S3 Bucket"
            }
          },
          {
            "name": "ecr_repository",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ],
              "title": "Ecr Repository"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/training/models/export": {
      "post": {
        "tags": [
          "training"
        ],
        "summary": "Export Model From Training Job",
        "description": "Export trained model from training job in production-compatible format.\n\nThis endpoint exports a trained model from a completed training job,\ncreating a production-ready model file with metadata and validation.",
        "operationId": "export_model_from_training_job_api_v1_training_models_export_post",
        "parameters": [
          {
            "name": "training_job_id",
            "in": "query",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Training Job Id"
            }
          },
          {
            "name": "age_group_min",
            "in": "query",
            "required": true,
            "schema": {
              "type": "number",
              "title": "Age Group Min"
            }
          },
          {
            "name": "age_group_max",
            "in": "query",
            "required": true,
            "schema": {
              "type": "number",
              "title": "Age Group Max"
            }
          },
          {
            "name": "export_format",
            "in": "query",
            "required": false,
            "schema": {
              "type": "string",
              "default": "pytorch",
              "title": "Export Format"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/training/models/exports": {
      "get": {
        "tags": [
          "training"
        ],
        "summary": "List Exported Models",
        "description": "List all exported models with their metadata.\n\nThis endpoint returns a list of all models that have been exported,\nincluding their metadata, export timestamps, and file information.",
        "operationId": "list_exported_models_api_v1_training_models_exports_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/v1/training/models/validate": {
      "post": {
        "tags": [
          "training"
        ],
        "summary": "Validate Exported Model",
        "description": "Validate exported model for compatibility and integrity.\n\nThis endpoint performs comprehensive validation of an exported model,\nchecking file integrity, compatibility, and performance metrics.",
        "operationId": "validate_exported_model_api_v1_training_models_validate_post",
        "parameters": [
          {
            "name": "model_id",
            "in": "query",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Model Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/training/models/deploy": {
      "post": {
        "tags": [
          "training"
        ],
        "summary": "Deploy Exported Model",
        "description": "Deploy exported model to production environment.\n\nThis endpoint deploys an exported model to the production system,\nmaking it available for anomaly detection in the specified age group.",
        "operationId": "deploy_exported_model_api_v1_training_models_deploy_post",
        "parameters": [
          {
            "name": "model_export_path",
            "in": "query",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Model Export Path"
            }
          },
          {
            "name": "age_group_min",
            "in": "query",
            "required": true,
            "schema": {
              "type": "number",
              "title": "Age Group Min"
            }
          },
          {
            "name": "age_group_max",
            "in": "query",
            "required": true,
            "schema": {
              "type": "number",
              "title": "Age Group Max"
            }
          },
          {
            "name": "replace_existing",
            "in": "query",
            "required": false,
            "schema": {
              "type": "boolean",
              "default": false,
              "title": "Replace Existing"
            }
          },
          {
            "name": "validate_before_deployment",
            "in": "query",
            "required": false,
            "schema": {
              "type": "boolean",
              "default": true,
              "title": "Validate Before Deployment"
            }
          },
          {
            "name": "backup_existing",
            "in": "query",
            "required": false,
            "schema": {
              "type": "boolean",
              "default": true,
              "title": "Backup Existing"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/training/models/deployed": {
      "get": {
        "tags": [
          "training"
        ],
        "summary": "List Deployed Models",
        "description": "List all deployed models in production.\n\nThis endpoint returns information about all models currently\ndeployed and active in the production system.",
        "operationId": "list_deployed_models_api_v1_training_models_deployed_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/v1/training/models/{model_id}/undeploy": {
      "post": {
        "tags": [
          "training"
        ],
        "summary": "Undeploy Model",
        "description": "Undeploy (deactivate) a deployed model.\n\nThis endpoint deactivates a deployed model, removing it from\nactive use in the production system.",
        "operationId": "undeploy_model_api_v1_training_models__model_id__undeploy_post",
        "parameters": [
          {
            "name": "model_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Model Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/config/": {
      "get": {
        "tags": [
          "configuration"
        ],
        "summary": "Get Config",
        "description": "Get current system configuration.\n\nThis endpoint returns the current system configuration including\nmodel settings, threshold parameters, and age grouping strategy.",
        "operationId": "get_config_api_v1_config__get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SystemConfigurationResponse"
                }
              }
            }
          }
        }
      },
      "put": {
        "tags": [
          "configuration"
        ],
        "summary": "Update Config",
        "description": "Update system configuration.\n\nThis endpoint updates various system configuration settings\nincluding thresholds and age grouping parameters.",
        "operationId": "update_config_api_v1_config__put",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ConfigurationUpdateRequest"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SuccessResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/config/threshold": {
      "put": {
        "tags": [
          "configuration"
        ],
        "summary": "Update Threshold Settings",
        "description": "Update global threshold settings.\n\nThis endpoint recalculates thresholds for all active models\nusing the specified percentile value from the request body.",
        "operationId": "update_threshold_settings_api_v1_config_threshold_put",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ThresholdUpdateRequest"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SuccessResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/config/age-grouping": {
      "put": {
        "tags": [
          "configuration"
        ],
        "summary": "Update Age Grouping",
        "description": "Modify age grouping strategy.\n\nThis endpoint updates the age grouping configuration and can\noptionally trigger recreation of age group models.",
        "operationId": "update_age_grouping_api_v1_config_age_grouping_put",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ConfigurationUpdateRequest"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SuccessResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/config/health": {
      "get": {
        "tags": [
          "configuration"
        ],
        "summary": "Health Check",
        "description": "System health check endpoint.\n\nThis endpoint provides information about the health and status\nof various system components.",
        "operationId": "health_check_api_v1_config_health_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HealthCheckResponse"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/config/stats": {
      "get": {
        "tags": [
          "configuration"
        ],
        "summary": "Get System Stats",
        "description": "Get comprehensive system statistics.\n\nThis endpoint provides detailed statistics about the system\nincluding data distribution, model performance, and usage metrics.",
        "operationId": "get_system_stats_api_v1_config_stats_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/v1/config/subjects": {
      "get": {
        "tags": [
          "configuration"
        ],
        "summary": "Get Supported Subject Categories",
        "description": "Get list of supported subject categories.\n\nThis endpoint returns all supported subject categories that can be used\nwhen uploading drawings, along with usage statistics.",
        "operationId": "get_supported_subject_categories_api_v1_config_subjects_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/v1/config/subjects/statistics": {
      "get": {
        "tags": [
          "configuration"
        ],
        "summary": "Get Subject Specific Statistics",
        "description": "Get subject-specific statistics and analysis data.\n\nThis endpoint provides detailed statistics about drawings and analyses\nfor specific subject categories or overall subject-related metrics.",
        "operationId": "get_subject_specific_statistics_api_v1_config_subjects_statistics_get",
        "parameters": [
          {
            "name": "subject",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ],
              "title": "Subject"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/config/models/subject-aware": {
      "get": {
        "tags": [
          "configuration"
        ],
        "summary": "Get Subject Aware Model Status",
        "description": "Get status of subject-aware model capabilities.\n\nThis endpoint provides information about the current subject-aware\nmodeling capabilities and model status.",
        "operationId": "get_subject_aware_model_status_api_v1_config_models_subject_aware_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/v1/config/reset": {
      "post": {
        "tags": [
          "configuration"
        ],
        "summary": "Reset System",
        "description": "Reset system configuration and models.\n\nWARNING: This endpoint deactivates all models and clears caches.\nUse with caution in production environments.",
        "operationId": "reset_system_api_v1_config_reset_post",
        "parameters": [
          {
            "name": "confirm",
            "in": "query",
            "required": false,
            "schema": {
              "type": "boolean",
              "default": false,
              "title": "Confirm"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/documentation/status": {
      "get": {
        "tags": [
          "documentation"
        ],
        "summary": "Get Documentation Status",
        "description": "Get current documentation generation status.\n\nReturns real-time status of documentation generation including progress,\ncurrent task, and any errors or warnings.",
        "operationId": "get_documentation_status_api_v1_documentation_status_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/DocumentationStatus"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/documentation/metrics": {
      "get": {
        "tags": [
          "documentation"
        ],
        "summary": "Get Documentation Metrics",
        "description": "Get comprehensive documentation metrics.\n\nReturns metrics about documentation files, generation history,\nsuccess rates, and validation status.",
        "operationId": "get_documentation_metrics_api_v1_documentation_metrics_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/DocumentationMetrics"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/documentation/generate": {
      "post": {
        "tags": [
          "documentation"
        ],
        "summary": "Generate Documentation",
        "description": "Trigger documentation generation.\n\nStarts documentation generation process in the background.\nUse the status endpoint to monitor progress.",
        "operationId": "generate_documentation_api_v1_documentation_generate_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/GenerationRequest"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/GenerationResult"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/documentation/generate/sync": {
      "post": {
        "tags": [
          "documentation"
        ],
        "summary": "Generate Documentation Sync",
        "description": "Generate documentation synchronously.\n\nRuns documentation generation and waits for completion.\nUse this for smaller generation tasks or when immediate results are needed.",
        "operationId": "generate_documentation_sync_api_v1_documentation_generate_sync_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/GenerationRequest"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/GenerationResult"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/documentation/categories": {
      "get": {
        "tags": [
          "documentation"
        ],
        "summary": "Get Documentation Categories",
        "description": "Get available documentation categories.\n\nReturns list of available documentation categories that can be generated.",
        "operationId": "get_documentation_categories_api_v1_documentation_categories_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/v1/documentation/files": {
      "get": {
        "tags": [
          "documentation"
        ],
        "summary": "Get Documentation Files",
        "description": "Get list of documentation files with metadata.\n\nReturns comprehensive list of documentation files with metadata,\nfiltering, and search capabilities.",
        "operationId": "get_documentation_files_api_v1_documentation_files_get",
        "parameters": [
          {
            "name": "category",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ],
              "description": "Filter by category",
              "title": "Category"
            },
            "description": "Filter by category"
          },
          {
            "name": "search",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ],
              "description": "Search in file names and content",
              "title": "Search"
            },
            "description": "Search in file names and content"
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/documentation/cache": {
      "delete": {
        "tags": [
          "documentation"
        ],
        "summary": "Clear Documentation Cache",
        "description": "Clear documentation generation cache.\n\nForces regeneration of all documentation by clearing the cache.",
        "operationId": "clear_documentation_cache_api_v1_documentation_cache_delete",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/v1/documentation/validation": {
      "get": {
        "tags": [
          "documentation"
        ],
        "summary": "Get Validation Status",
        "description": "Get comprehensive validation status for all documentation.\n\nReturns detailed validation results including errors, warnings,\nand quality metrics.",
        "operationId": "get_validation_status_api_v1_documentation_validation_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/v1/documentation/validate": {
      "post": {
        "tags": [
          "documentation"
        ],
        "summary": "Validate Documentation",
        "description": "Run validation on documentation.\n\nValidates documentation for technical accuracy, link integrity,\naccessibility compliance, and formatting consistency.",
        "operationId": "validate_documentation_api_v1_documentation_validate_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "anyOf": [
                  {
                    "items": {
                      "type": "string"
                    },
                    "type": "array"
                  },
                  {
                    "type": "null"
                  }
                ],
                "title": "Categories"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/documentation/preview/{category}": {
      "get": {
        "tags": [
          "documentation"
        ],
        "summary": "Preview Documentation Changes",
        "description": "Preview documentation changes before generation.\n\nShows what would be generated for a specific category or file\nwithout actually writing the files.",
        "operationId": "preview_documentation_changes_api_v1_documentation_preview__category__get",
        "parameters": [
          {
            "name": "category",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Category"
            }
          },
          {
            "name": "file_path",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ],
              "description": "Specific file to preview",
              "title": "File Path"
            },
            "description": "Specific file to preview"
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/documentation/batch/generate": {
      "post": {
        "tags": [
          "documentation"
        ],
        "summary": "Batch Generate Documentation",
        "description": "Batch generate multiple documentation categories with scheduling.\n\nAllows generating multiple categories in sequence with different\nconfigurations for each category.",
        "operationId": "batch_generate_documentation_api_v1_documentation_batch_generate_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "additionalProperties": true,
                "type": "object",
                "title": "Request"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/documentation/batch/validate": {
      "post": {
        "tags": [
          "documentation"
        ],
        "summary": "Batch Validate Documentation",
        "description": "Batch validate multiple documentation categories.\n\nRuns validation on multiple categories in parallel for faster processing.",
        "operationId": "batch_validate_documentation_api_v1_documentation_batch_validate_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "items": {
                  "type": "string"
                },
                "type": "array",
                "title": "Categories"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/documentation/schedule": {
      "get": {
        "tags": [
          "documentation"
        ],
        "summary": "Get Generation Schedule",
        "description": "Get current generation schedule and queue.\n\nReturns information about scheduled and queued generation tasks.",
        "operationId": "get_generation_schedule_api_v1_documentation_schedule_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      },
      "post": {
        "tags": [
          "documentation"
        ],
        "summary": "Schedule Generation",
        "description": "Schedule documentation generation for later execution.\n\nAllows scheduling generation tasks for specific times or intervals.",
        "operationId": "schedule_generation_api_v1_documentation_schedule_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "additionalProperties": true,
                "type": "object",
                "title": "Request"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/documentation/search": {
      "post": {
        "tags": [
          "documentation"
        ],
        "summary": "Search Documentation",
        "description": "Search documentation with advanced filtering and faceting.\n\nProvides full-text search across all documentation with relevance scoring,\nfaceted filtering, and intelligent suggestions.",
        "operationId": "search_documentation_api_v1_documentation_search_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/SearchRequest"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SearchResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/documentation/search/suggestions": {
      "get": {
        "tags": [
          "documentation"
        ],
        "summary": "Get Search Suggestions",
        "description": "Get search suggestions for autocomplete.\n\nProvides intelligent search suggestions based on indexed content\nand common search patterns.",
        "operationId": "get_search_suggestions_api_v1_documentation_search_suggestions_get",
        "parameters": [
          {
            "name": "query",
            "in": "query",
            "required": true,
            "schema": {
              "type": "string",
              "description": "Partial query for suggestions",
              "title": "Query"
            },
            "description": "Partial query for suggestions"
          },
          {
            "name": "limit",
            "in": "query",
            "required": false,
            "schema": {
              "type": "integer",
              "description": "Maximum number of suggestions",
              "default": 10,
              "title": "Limit"
            },
            "description": "Maximum number of suggestions"
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/documentation/search/statistics": {
      "get": {
        "tags": [
          "documentation"
        ],
        "summary": "Get Search Statistics",
        "description": "Get search index statistics.\n\nReturns comprehensive statistics about the search index including\ndocument counts, index size, and performance metrics.",
        "operationId": "get_search_statistics_api_v1_documentation_search_statistics_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/v1/documentation/search/index": {
      "post": {
        "tags": [
          "documentation"
        ],
        "summary": "Rebuild Search Index",
        "description": "Rebuild the search index.\n\nRebuilds the search index from all documentation files.\nUse force=true to completely rebuild the index.",
        "operationId": "rebuild_search_index_api_v1_documentation_search_index_post",
        "parameters": [
          {
            "name": "force",
            "in": "query",
            "required": false,
            "schema": {
              "type": "boolean",
              "description": "Force complete reindexing",
              "default": false,
              "title": "Force"
            },
            "description": "Force complete reindexing"
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/documentation/navigation/{document_id}": {
      "get": {
        "tags": [
          "documentation"
        ],
        "summary": "Get Navigation Context",
        "description": "Get navigation context for a document.\n\nReturns comprehensive navigation context including breadcrumbs,\ncross-references, related content, and sequential navigation.",
        "operationId": "get_navigation_context_api_v1_documentation_navigation__document_id__get",
        "parameters": [
          {
            "name": "document_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Document Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/documentation/navigation/sitemap": {
      "get": {
        "tags": [
          "documentation"
        ],
        "summary": "Get Sitemap",
        "description": "Get complete documentation sitemap.\n\nReturns hierarchical sitemap of all documentation organized by type\nand category with metadata.",
        "operationId": "get_sitemap_api_v1_documentation_navigation_sitemap_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/v1/documentation/navigation/cross-references": {
      "get": {
        "tags": [
          "documentation"
        ],
        "summary": "Get Cross Reference Report",
        "description": "Get cross-reference analysis report.\n\nReturns comprehensive analysis of cross-references including\nbroken links, most referenced documents, and orphaned content.",
        "operationId": "get_cross_reference_report_api_v1_documentation_navigation_cross_references_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/v1/documentation/navigation/rebuild": {
      "post": {
        "tags": [
          "documentation"
        ],
        "summary": "Rebuild Navigation Structure",
        "description": "Rebuild navigation structure.\n\nRebuilds the navigation structure and cross-reference index\nfrom all documentation files.",
        "operationId": "rebuild_navigation_structure_api_v1_documentation_navigation_rebuild_post",
        "parameters": [
          {
            "name": "force",
            "in": "query",
            "required": false,
            "schema": {
              "type": "boolean",
              "description": "Force complete rebuild",
              "default": false,
              "title": "Force"
            },
            "description": "Force complete rebuild"
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/metrics/usage": {
      "get": {
        "tags": [
          "metrics"
        ],
        "summary": "Get Usage Metrics",
        "description": "Get comprehensive usage metrics for the dashboard.\n\nReturns metrics including:\n- Total analyses and drawings\n- Time-based analysis counts (daily, weekly, monthly)\n- Active user sessions and geographic distribution\n- System health and performance metrics\n- Processing time statistics",
        "operationId": "get_usage_metrics_api_v1_metrics_usage_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "additionalProperties": true,
                  "type": "object",
                  "title": "Response Get Usage Metrics Api V1 Metrics Usage Get"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/metrics/health": {
      "get": {
        "tags": [
          "metrics"
        ],
        "summary": "Get System Health",
        "description": "Get system health metrics including uptime, error rates, and resource usage.",
        "operationId": "get_system_health_api_v1_metrics_health_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "additionalProperties": true,
                  "type": "object",
                  "title": "Response Get System Health Api V1 Metrics Health Get"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/metrics/sessions": {
      "get": {
        "tags": [
          "metrics"
        ],
        "summary": "Get Session Metrics",
        "description": "Get current user session metrics and geographic distribution.",
        "operationId": "get_session_metrics_api_v1_metrics_sessions_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "additionalProperties": true,
                  "type": "object",
                  "title": "Response Get Session Metrics Api V1 Metrics Sessions Get"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/metrics/performance": {
      "get": {
        "tags": [
          "metrics"
        ],
        "summary": "Get Performance Metrics",
        "description": "Get detailed performance metrics including processing times and system resources.",
        "operationId": "get_performance_metrics_api_v1_metrics_performance_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "additionalProperties": true,
                  "type": "object",
                  "title": "Response Get Performance Metrics Api V1 Metrics Performance Get"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/metrics/session/start": {
      "post": {
        "tags": [
          "metrics"
        ],
        "summary": "Start User Session",
        "description": "Manually start a user session (alternative to automatic detection).\n\nRequest body should contain:\n- ip_address: Client IP address\n- user_agent: User agent string",
        "operationId": "start_user_session_api_v1_metrics_session_start_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "additionalProperties": {
                  "type": "string"
                },
                "type": "object",
                "title": "Request Info"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "additionalProperties": true,
                  "type": "object",
                  "title": "Response Start User Session Api V1 Metrics Session Start Post"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/metrics/session/{session_id}/end": {
      "post": {
        "tags": [
          "metrics"
        ],
        "summary": "End User Session",
        "description": "Manually end a user session.",
        "operationId": "end_user_session_api_v1_metrics_session__session_id__end_post",
        "parameters": [
          {
            "name": "session_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Session Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "additionalProperties": true,
                  "title": "Response End User Session Api V1 Metrics Session  Session Id  End Post"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/metrics/cloudwatch/status": {
      "get": {
        "tags": [
          "metrics"
        ],
        "summary": "Get Cloudwatch Status",
        "description": "Get CloudWatch integration status and configuration.",
        "operationId": "get_cloudwatch_status_api_v1_metrics_cloudwatch_status_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "additionalProperties": true,
                  "type": "object",
                  "title": "Response Get Cloudwatch Status Api V1 Metrics Cloudwatch Status Get"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/demo/": {
      "get": {
        "tags": [
          "demo"
        ],
        "summary": "Get Demo Page",
        "description": "Get the complete demo page with all content.\n\nReturns:\n    HTML response with complete demo page content",
        "operationId": "get_demo_page_api_v1_demo__get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "text/html": {
                "schema": {
                  "type": "string"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/demo/samples": {
      "get": {
        "tags": [
          "demo"
        ],
        "summary": "Get Demo Samples",
        "description": "Get all demo samples with analysis results.\n\nReturns:\n    List of demo samples with complete analysis data",
        "operationId": "get_demo_samples_api_v1_demo_samples_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SuccessResponse"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/demo/samples/{sample_id}": {
      "get": {
        "tags": [
          "demo"
        ],
        "summary": "Get Demo Sample",
        "description": "Get a specific demo sample by ID.\n\nArgs:\n    sample_id: ID of the demo sample\n\nReturns:\n    Demo sample with complete analysis data",
        "operationId": "get_demo_sample_api_v1_demo_samples__sample_id__get",
        "parameters": [
          {
            "name": "sample_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Sample Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SuccessResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/demo/project-info": {
      "get": {
        "tags": [
          "demo"
        ],
        "summary": "Get Project Info",
        "description": "Get comprehensive project information for demo page.\n\nReturns:\n    Project description with technical details and features",
        "operationId": "get_project_info_api_v1_demo_project_info_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SuccessResponse"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/demo/disclaimer": {
      "get": {
        "tags": [
          "demo"
        ],
        "summary": "Get Medical Disclaimer",
        "description": "Get medical disclaimer and warnings for demo content.\n\nReturns:\n    Medical disclaimer with all required warnings",
        "operationId": "get_medical_disclaimer_api_v1_demo_disclaimer_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SuccessResponse"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/demo/technical-links": {
      "get": {
        "tags": [
          "demo"
        ],
        "summary": "Get Technical Links",
        "description": "Get technical links and documentation references.\n\nReturns:\n    Technical links including GitHub repository and documentation",
        "operationId": "get_technical_links_api_v1_demo_technical_links_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SuccessResponse"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/demo/statistics": {
      "get": {
        "tags": [
          "demo"
        ],
        "summary": "Get Demo Statistics",
        "description": "Get demo-specific statistics and metrics.\n\nReturns:\n    Demo statistics including sample counts and distributions",
        "operationId": "get_demo_statistics_api_v1_demo_statistics_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SuccessResponse"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/files/s3/{file_path}": {
      "get": {
        "tags": [
          "files"
        ],
        "summary": "Serve S3 File",
        "description": "Serve a file from S3 storage.\n\nThis endpoint downloads files from S3 and serves them with proper caching headers.\nThis avoids presigned URL expiration issues and allows CloudFront to cache responses.\n\nArgs:\n    file_path: S3 key path (e.g., \"drawings/20240108_123456_abc123.png\")\n\nReturns:\n    File response with caching headers",
        "operationId": "serve_s3_file_api_v1_files_s3__file_path__get",
        "parameters": [
          {
            "name": "file_path",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "File Path"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      },
      "head": {
        "tags": [
          "files"
        ],
        "summary": "Check S3 File",
        "description": "Check if a file exists in S3 storage.\n\nArgs:\n    file_path: S3 key path\n\nReturns:\n    Empty response with appropriate status code",
        "operationId": "check_s3_file_api_v1_files_s3__file_path__head",
        "parameters": [
          {
            "name": "file_path",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "File Path"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/files/markdown": {
      "get": {
        "tags": [
          "files"
        ],
        "summary": "Serve Markdown File",
        "description": "Serve a markdown file from the local filesystem.\n\nArgs:\n    path: Relative path to markdown file (e.g., \"tmp_files/analysis.md\")\n\nReturns:\n    Markdown file content as plain text",
        "operationId": "serve_markdown_file_api_v1_files_markdown_get",
        "parameters": [
          {
            "name": "path",
            "in": "query",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Path"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/database/backup": {
      "post": {
        "tags": [
          "database"
        ],
        "summary": "Create Database Backup",
        "description": "Create a database backup with optional S3 upload.\n\n- **upload_to_s3**: Whether to upload to S3 (defaults to environment setting)\n- **include_files**: Whether to include uploaded files and static content",
        "operationId": "create_database_backup_api_v1_database_backup_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/BackupRequest"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "additionalProperties": true,
                  "type": "object",
                  "title": "Response Create Database Backup Api V1 Database Backup Post"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/database/migrate": {
      "post": {
        "tags": [
          "database"
        ],
        "summary": "Run Database Migration",
        "description": "Run database migrations to the specified revision.\n\n- **target_revision**: Target migration revision (defaults to \"head\")",
        "operationId": "run_database_migration_api_v1_database_migrate_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/MigrationRequest"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "additionalProperties": true,
                  "type": "object",
                  "title": "Response Run Database Migration Api V1 Database Migrate Post"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/database/migration-info": {
      "get": {
        "tags": [
          "database"
        ],
        "summary": "Get Migration Info",
        "description": "Get current database migration information.",
        "operationId": "get_migration_info_api_v1_database_migration_info_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "additionalProperties": true,
                  "type": "object",
                  "title": "Response Get Migration Info Api V1 Database Migration Info Get"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/database/validate-consistency": {
      "post": {
        "tags": [
          "database"
        ],
        "summary": "Validate Cross Environment Consistency",
        "description": "Validate database schema consistency across environments.\n\n- **other_db_url**: Database URL of the other environment to compare",
        "operationId": "validate_cross_environment_consistency_api_v1_database_validate_consistency_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ConsistencyCheckRequest"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "additionalProperties": true,
                  "type": "object",
                  "title": "Response Validate Cross Environment Consistency Api V1 Database Validate Consistency Post"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/database/backup-list": {
      "get": {
        "tags": [
          "database"
        ],
        "summary": "List Backups",
        "description": "Get list of available database backups.",
        "operationId": "list_backups_api_v1_database_backup_list_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "additionalProperties": true,
                  "type": "object",
                  "title": "Response List Backups Api V1 Database Backup List Get"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/database/schedule-backups": {
      "post": {
        "tags": [
          "database"
        ],
        "summary": "Schedule Automated Backups",
        "description": "Schedule automated database backups.\n\n- **interval_hours**: Backup interval in hours (default: 6)",
        "operationId": "schedule_automated_backups_api_v1_database_schedule_backups_post",
        "parameters": [
          {
            "name": "interval_hours",
            "in": "query",
            "required": false,
            "schema": {
              "type": "integer",
              "default": 6,
              "title": "Interval Hours"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "additionalProperties": true,
                  "title": "Response Schedule Automated Backups Api V1 Database Schedule Backups Post"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/database/consistency-check": {
      "post": {
        "tags": [
          "database"
        ],
        "summary": "Run Consistency Check",
        "description": "Run database consistency validation.",
        "operationId": "run_consistency_check_api_v1_database_consistency_check_post",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "additionalProperties": true,
                  "type": "object",
                  "title": "Response Run Consistency Check Api V1 Database Consistency Check Post"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/security/status": {
      "get": {
        "tags": [
          "security"
        ],
        "summary": "Get Security Status",
        "description": "Get current security service status and configuration.\n\nReturns information about security service initialization,\nAWS client availability, and current security policy.",
        "operationId": "get_security_status_api_v1_security_status_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "additionalProperties": true,
                  "type": "object",
                  "title": "Response Get Security Status Api V1 Security Status Get"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/security/validate/iam-role": {
      "post": {
        "tags": [
          "security"
        ],
        "summary": "Validate Iam Role",
        "description": "Validate IAM role for least-privilege compliance.\n\nChecks the specified IAM role for overly broad permissions,\ndangerous policy attachments, and compliance with security best practices.",
        "operationId": "validate_iam_role_api_v1_security_validate_iam_role_post",
        "parameters": [
          {
            "name": "role_arn",
            "in": "query",
            "required": true,
            "schema": {
              "type": "string",
              "description": "IAM role ARN to validate",
              "title": "Role Arn"
            },
            "description": "IAM role ARN to validate"
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SecurityValidationResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/security/validate/s3-bucket": {
      "post": {
        "tags": [
          "security"
        ],
        "summary": "Validate S3 Bucket",
        "description": "Validate S3 bucket encryption and security configuration.\n\nChecks the specified S3 bucket for proper encryption configuration,\npublic access blocks, and security compliance.",
        "operationId": "validate_s3_bucket_api_v1_security_validate_s3_bucket_post",
        "parameters": [
          {
            "name": "bucket_name",
            "in": "query",
            "required": true,
            "schema": {
              "type": "string",
              "description": "S3 bucket name to validate",
              "title": "Bucket Name"
            },
            "description": "S3 bucket name to validate"
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SecurityValidationResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/security/validate/security-groups": {
      "post": {
        "tags": [
          "security"
        ],
        "summary": "Validate Security Groups",
        "description": "Validate security group configurations for minimal exposure.\n\nChecks the specified security groups for overly permissive rules,\nopen ports, and compliance with network security best practices.",
        "operationId": "validate_security_groups_api_v1_security_validate_security_groups_post",
        "parameters": [
          {
            "name": "security_group_ids",
            "in": "query",
            "required": true,
            "schema": {
              "type": "array",
              "items": {
                "type": "string"
              },
              "description": "Security group IDs to validate",
              "title": "Security Group Ids"
            },
            "description": "Security group IDs to validate"
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SecurityValidationResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/security/validate/vpc": {
      "post": {
        "tags": [
          "security"
        ],
        "summary": "Validate Vpc",
        "description": "Validate VPC configuration for security compliance.\n\nChecks the specified VPC for proper subnet configuration,\nprivate subnet isolation, and network security best practices.",
        "operationId": "validate_vpc_api_v1_security_validate_vpc_post",
        "parameters": [
          {
            "name": "vpc_id",
            "in": "query",
            "required": true,
            "schema": {
              "type": "string",
              "description": "VPC ID to validate",
              "title": "Vpc Id"
            },
            "description": "VPC ID to validate"
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SecurityValidationResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/security/validate/encryption-in-transit": {
      "get": {
        "tags": [
          "security"
        ],
        "summary": "Validate Encryption In Transit",
        "description": "Validate encryption in transit configuration.\n\nChecks the current deployment for proper HTTPS enforcement,\nTLS configuration, and secure communication protocols.",
        "operationId": "validate_encryption_in_transit_api_v1_security_validate_encryption_in_transit_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SecurityValidationResponse"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/security/audit/comprehensive": {
      "post": {
        "tags": [
          "security"
        ],
        "summary": "Comprehensive Security Audit",
        "description": "Perform comprehensive security audit of AWS resources.\n\nValidates all specified AWS resources for security compliance,\nincluding IAM roles, S3 buckets, security groups, VPC configuration,\nand encryption settings.",
        "operationId": "comprehensive_security_audit_api_v1_security_audit_comprehensive_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/SecurityAuditRequest"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SecurityAuditResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/security/compliance/report": {
      "get": {
        "tags": [
          "security"
        ],
        "summary": "Get Compliance Report",
        "description": "Generate security compliance report.\n\nCreates a detailed compliance report based on current security\nconfiguration and validation results.",
        "operationId": "get_compliance_report_api_v1_security_compliance_report_get",
        "parameters": [
          {
            "name": "format",
            "in": "query",
            "required": false,
            "schema": {
              "type": "string",
              "description": "Report format: json, csv, or html",
              "default": "json",
              "title": "Format"
            },
            "description": "Report format: json, csv, or html"
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/cost-optimization/estimate": {
      "get": {
        "tags": [
          "cost-optimization"
        ],
        "summary": "Get Cost Estimate",
        "description": "Get estimated monthly costs for optimized AWS resources.\n\nReturns cost breakdown and compliance status for the production deployment.",
        "operationId": "get_cost_estimate_api_v1_cost_optimization_estimate_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/CostEstimateResponse"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/cost-optimization/optimization": {
      "get": {
        "tags": [
          "cost-optimization"
        ],
        "summary": "Get Cost Optimization",
        "description": "Get cost optimization configurations and recommendations.\n\nReturns optimized configurations for ECS Fargate, S3, and CloudFront.",
        "operationId": "get_cost_optimization_api_v1_cost_optimization_optimization_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/CostOptimizationResponse"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/cost-optimization/compliance": {
      "get": {
        "tags": [
          "cost-optimization"
        ],
        "summary": "Validate Cost Compliance",
        "description": "Validate cost compliance against budget requirements.\n\nReturns compliance status and detailed cost analysis.",
        "operationId": "validate_cost_compliance_api_v1_cost_optimization_compliance_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/CostComplianceResponse"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/cost-optimization/apply-s3-lifecycle/{bucket_name}": {
      "post": {
        "tags": [
          "cost-optimization"
        ],
        "summary": "Apply S3 Lifecycle Optimization",
        "description": "Apply S3 lifecycle optimization to a specific bucket.\n\nArgs:\n    bucket_name: Name of the S3 bucket to optimize\n\nReturns:\n    Success status of the lifecycle policy application",
        "operationId": "apply_s3_lifecycle_optimization_api_v1_cost_optimization_apply_s3_lifecycle__bucket_name__post",
        "parameters": [
          {
            "name": "bucket_name",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Bucket Name"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/cost-optimization/setup-monitoring": {
      "post": {
        "tags": [
          "cost-optimization"
        ],
        "summary": "Setup Cost Monitoring",
        "description": "Set up cost monitoring and budget alerts.\n\nReturns:\n    Success status of the cost monitoring setup",
        "operationId": "setup_cost_monitoring_api_v1_cost_optimization_setup_monitoring_post",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/v1/cost-optimization/config": {
      "get": {
        "tags": [
          "cost-optimization"
        ],
        "summary": "Get Cost Optimization Config",
        "description": "Get current cost optimization configuration.\n\nReturns:\n    Current cost optimization settings",
        "operationId": "get_cost_optimization_config_api_v1_cost_optimization_config_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/v1/health": {
      "get": {
        "tags": [
          "health"
        ],
        "summary": "Basic health check",
        "description": "Basic health check endpoint.",
        "operationId": "basic_health_check_api_v1_health_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/v1/health/detailed": {
      "get": {
        "tags": [
          "health"
        ],
        "summary": "Detailed health check",
        "description": "Detailed health check with all system components.",
        "operationId": "detailed_health_check_api_v1_health_detailed_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/v1/health/component/{component_name}": {
      "get": {
        "tags": [
          "health"
        ],
        "summary": "Component-specific health check",
        "description": "Get health status for a specific component.",
        "operationId": "component_health_check_api_v1_health_component__component_name__get",
        "parameters": [
          {
            "name": "component_name",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Component Name"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/metrics": {
      "get": {
        "tags": [
          "health"
        ],
        "summary": "System metrics",
        "description": "Get current system metrics.",
        "operationId": "get_system_metrics_api_v1_metrics_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/v1/metrics/history": {
      "get": {
        "tags": [
          "health"
        ],
        "summary": "Historical metrics",
        "description": "Get historical system metrics.",
        "operationId": "get_metrics_history_api_v1_metrics_history_get",
        "parameters": [
          {
            "name": "hours",
            "in": "query",
            "required": false,
            "schema": {
              "type": "integer",
              "maximum": 24,
              "minimum": 1,
              "description": "Hours of history to retrieve",
              "default": 1,
              "title": "Hours"
            },
            "description": "Hours of history to retrieve"
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/alerts": {
      "get": {
        "tags": [
          "health"
        ],
        "summary": "Current system alerts",
        "description": "Get current system alerts.",
        "operationId": "get_current_alerts_api_v1_alerts_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/v1/alerts/thresholds": {
      "post": {
        "tags": [
          "health"
        ],
        "summary": "Update alert thresholds",
        "description": "Update system alert thresholds.",
        "operationId": "update_alert_thresholds_api_v1_alerts_thresholds_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "additionalProperties": {
                  "type": "number"
                },
                "type": "object",
                "title": "Thresholds"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/status": {
      "get": {
        "tags": [
          "health"
        ],
        "summary": "Overall system status",
        "description": "Get overall system status summary.",
        "operationId": "get_system_status_api_v1_status_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/v1/backup/full": {
      "post": {
        "tags": [
          "backup"
        ],
        "summary": "Create full system backup",
        "description": "Create a full system backup including database and files.",
        "operationId": "create_full_backup_api_v1_backup_full_post",
        "parameters": [
          {
            "name": "include_files",
            "in": "query",
            "required": false,
            "schema": {
              "type": "boolean",
              "description": "Include uploaded files and generated content",
              "default": true,
              "title": "Include Files"
            },
            "description": "Include uploaded files and generated content"
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/backup/database": {
      "post": {
        "tags": [
          "backup"
        ],
        "summary": "Create database backup",
        "description": "Create a database-only backup.",
        "operationId": "create_database_backup_api_v1_backup_database_post",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/v1/backup/list": {
      "get": {
        "tags": [
          "backup"
        ],
        "summary": "List available backups",
        "description": "Get list of available backup files.",
        "operationId": "list_backups_api_v1_backup_list_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/v1/backup/download/{backup_name}": {
      "get": {
        "tags": [
          "backup"
        ],
        "summary": "Download backup file",
        "description": "Download a specific backup file.",
        "operationId": "download_backup_api_v1_backup_download__backup_name__get",
        "parameters": [
          {
            "name": "backup_name",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Backup Name"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/backup/restore": {
      "post": {
        "tags": [
          "backup"
        ],
        "summary": "Restore from backup",
        "description": "Restore system from a backup file.",
        "operationId": "restore_from_backup_api_v1_backup_restore_post",
        "parameters": [
          {
            "name": "backup_name",
            "in": "query",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Backup Name"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/backup/upload": {
      "post": {
        "tags": [
          "backup"
        ],
        "summary": "Upload backup file",
        "description": "Upload a backup file for restoration.",
        "operationId": "upload_backup_api_v1_backup_upload_post",
        "requestBody": {
          "content": {
            "multipart/form-data": {
              "schema": {
                "$ref": "#/components/schemas/Body_upload_backup_api_v1_backup_upload_post"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/export": {
      "post": {
        "tags": [
          "backup"
        ],
        "summary": "Export system data",
        "description": "Export system data in specified format.",
        "operationId": "export_data_api_v1_export_post",
        "parameters": [
          {
            "name": "format",
            "in": "query",
            "required": false,
            "schema": {
              "type": "string",
              "pattern": "^(json|csv)$",
              "description": "Export format",
              "default": "json",
              "title": "Format"
            },
            "description": "Export format"
          },
          {
            "name": "include_embeddings",
            "in": "query",
            "required": false,
            "schema": {
              "type": "boolean",
              "description": "Include embedding vectors",
              "default": false,
              "title": "Include Embeddings"
            },
            "description": "Include embedding vectors"
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/export/download/{export_name}": {
      "get": {
        "tags": [
          "backup"
        ],
        "summary": "Download exported data",
        "description": "Download an exported data file.",
        "operationId": "download_export_api_v1_export_download__export_name__get",
        "parameters": [
          {
            "name": "export_name",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Export Name"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/backup/{backup_name}": {
      "delete": {
        "tags": [
          "backup"
        ],
        "summary": "Delete backup file",
        "description": "Delete a specific backup file.",
        "operationId": "delete_backup_api_v1_backup__backup_name__delete",
        "parameters": [
          {
            "name": "backup_name",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Backup Name"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/cleanup": {
      "post": {
        "tags": [
          "backup"
        ],
        "summary": "Clean up old backups",
        "description": "Clean up old backup files based on retention policy.",
        "operationId": "cleanup_backups_api_v1_cleanup_post",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/v1/storage/info": {
      "get": {
        "tags": [
          "backup"
        ],
        "summary": "Get storage information",
        "description": "Get information about storage usage and organization.",
        "operationId": "get_storage_info_api_v1_storage_info_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/v1/storage/cleanup": {
      "post": {
        "tags": [
          "backup"
        ],
        "summary": "Clean up temporary and orphaned files",
        "description": "Clean up temporary files and orphaned data.",
        "operationId": "cleanup_storage_api_v1_storage_cleanup_post",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/drawings/upload": {
      "post": {
        "tags": [
          "drawings"
        ],
        "summary": "Upload Drawing",
        "description": "Upload drawing with metadata.\n\nThis endpoint accepts multipart form data with an image file and metadata.\nThe image is validated, preprocessed, and stored along with the metadata.",
        "operationId": "upload_drawing_api_drawings_upload_post",
        "requestBody": {
          "content": {
            "multipart/form-data": {
              "schema": {
                "$ref": "#/components/schemas/Body_upload_drawing_api_drawings_upload_post"
              }
            }
          },
          "required": true
        },
        "responses": {
          "201": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/DrawingResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/drawings/upload/progress/{upload_id}": {
      "get": {
        "tags": [
          "drawings"
        ],
        "summary": "Get Upload Progress",
        "description": "Get upload progress for large file uploads.",
        "operationId": "get_upload_progress_api_drawings_upload_progress__upload_id__get",
        "parameters": [
          {
            "name": "upload_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Upload Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/drawings/{drawing_id}": {
      "get": {
        "tags": [
          "drawings"
        ],
        "summary": "Get Drawing",
        "description": "Retrieve drawing details by ID.",
        "operationId": "get_drawing_api_drawings__drawing_id__get",
        "parameters": [
          {
            "name": "drawing_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Drawing Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/DrawingResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      },
      "delete": {
        "tags": [
          "drawings"
        ],
        "summary": "Delete Drawing",
        "description": "Delete drawing and associated data.",
        "operationId": "delete_drawing_api_drawings__drawing_id__delete",
        "parameters": [
          {
            "name": "drawing_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Drawing Id"
            }
          }
        ],
        "responses": {
          "204": {
            "description": "Successful Response"
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/drawings/{drawing_id}/file": {
      "get": {
        "tags": [
          "drawings"
        ],
        "summary": "Get Drawing File",
        "description": "Retrieve the actual drawing file.",
        "operationId": "get_drawing_file_api_drawings__drawing_id__file_get",
        "parameters": [
          {
            "name": "drawing_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Drawing Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/drawings/": {
      "get": {
        "tags": [
          "drawings"
        ],
        "summary": "List Drawings",
        "description": "List drawings with optional filtering and pagination.",
        "operationId": "list_drawings_api_drawings__get",
        "parameters": [
          {
            "name": "age_min",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "number"
                },
                {
                  "type": "null"
                }
              ],
              "title": "Age Min"
            }
          },
          {
            "name": "age_max",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "number"
                },
                {
                  "type": "null"
                }
              ],
              "title": "Age Max"
            }
          },
          {
            "name": "subject",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ],
              "title": "Subject"
            }
          },
          {
            "name": "expert_label",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "$ref": "#/components/schemas/ExpertLabel"
                },
                {
                  "type": "null"
                }
              ],
              "title": "Expert Label"
            }
          },
          {
            "name": "page",
            "in": "query",
            "required": false,
            "schema": {
              "type": "integer",
              "default": 1,
              "title": "Page"
            }
          },
          {
            "name": "page_size",
            "in": "query",
            "required": false,
            "schema": {
              "type": "integer",
              "default": 20,
              "title": "Page Size"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/DrawingListResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/drawings/batch/upload": {
      "post": {
        "tags": [
          "drawings"
        ],
        "summary": "Batch Upload Drawings",
        "description": "Upload multiple drawings in batch.\n\nThis endpoint accepts multiple files and processes them in the background.\nReturns an upload ID for tracking progress.",
        "operationId": "batch_upload_drawings_api_drawings_batch_upload_post",
        "requestBody": {
          "content": {
            "multipart/form-data": {
              "schema": {
                "$ref": "#/components/schemas/Body_batch_upload_drawings_api_drawings_batch_upload_post"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/drawings/stats": {
      "get": {
        "tags": [
          "drawings"
        ],
        "summary": "Get Drawing Stats",
        "description": "Get statistics about stored drawings.",
        "operationId": "get_drawing_stats_api_drawings_stats_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/analysis/stats": {
      "get": {
        "tags": [
          "analysis"
        ],
        "summary": "Get Analysis Stats",
        "description": "Get dashboard statistics for analyses and drawings.\n\nThis endpoint provides comprehensive statistics for the dashboard\nincluding drawing counts, analysis results, and model status.",
        "operationId": "get_analysis_stats_api_analysis_stats_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/analysis/analyze/{drawing_id}": {
      "post": {
        "tags": [
          "analysis"
        ],
        "summary": "Analyze Drawing",
        "description": "Analyze specific drawing for anomalies.\n\nThis endpoint performs anomaly detection on a single drawing,\ngenerating embeddings, computing anomaly scores, and providing\ninterpretability results if the drawing is flagged as anomalous.",
        "operationId": "analyze_drawing_api_analysis_analyze__drawing_id__post",
        "parameters": [
          {
            "name": "drawing_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Drawing Id"
            }
          }
        ],
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/AnalysisRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/AnalysisResultResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/analysis/batch": {
      "post": {
        "tags": [
          "analysis"
        ],
        "summary": "Batch Analyze",
        "description": "Batch analyze multiple drawings.\n\nThis endpoint accepts a list of drawing IDs and processes them\nin the background, returning a batch ID for progress tracking.",
        "operationId": "batch_analyze_api_analysis_batch_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/BatchAnalysisRequest"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "additionalProperties": true,
                  "type": "object",
                  "title": "Response Batch Analyze Api Analysis Batch Post"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/analysis/batch/{batch_id}/progress": {
      "get": {
        "tags": [
          "analysis"
        ],
        "summary": "Get Batch Progress",
        "description": "Get progress of batch analysis.",
        "operationId": "get_batch_progress_api_analysis_batch__batch_id__progress_get",
        "parameters": [
          {
            "name": "batch_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Batch Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/BatchAnalysisResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/analysis/{analysis_id}": {
      "get": {
        "tags": [
          "analysis"
        ],
        "summary": "Get Analysis Result",
        "description": "Get analysis results by analysis ID.\n\nThis endpoint retrieves a complete analysis result including\nthe drawing information, anomaly analysis, and interpretability\nresults if available.",
        "operationId": "get_analysis_result_api_analysis__analysis_id__get",
        "parameters": [
          {
            "name": "analysis_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Analysis Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/AnalysisResultResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/analysis/embeddings/{drawing_id}": {
      "post": {
        "tags": [
          "analysis"
        ],
        "summary": "Generate Embedding",
        "description": "Generate embedding for a drawing without requiring a trained model.\n\nThis endpoint is used during the training phase to generate embeddings\nfor all drawings before training the autoencoder models.",
        "operationId": "generate_embedding_api_analysis_embeddings__drawing_id__post",
        "parameters": [
          {
            "name": "drawing_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Drawing Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/analysis/drawing/{drawing_id}": {
      "get": {
        "tags": [
          "analysis"
        ],
        "summary": "Get Drawing Analyses",
        "description": "Get all analyses for a specific drawing.\n\nThis endpoint returns the analysis history for a drawing,\nordered by most recent first.",
        "operationId": "get_drawing_analyses_api_analysis_drawing__drawing_id__get",
        "parameters": [
          {
            "name": "drawing_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Drawing Id"
            }
          },
          {
            "name": "limit",
            "in": "query",
            "required": false,
            "schema": {
              "type": "integer",
              "default": 10,
              "title": "Limit"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/AnalysisHistoryResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/interpretability/{analysis_id}/interactive": {
      "get": {
        "tags": [
          "interpretability"
        ],
        "summary": "Get Interactive Interpretability",
        "description": "Get interactive saliency data with hoverable regions and click explanations.\n\nThis endpoint provides enhanced interpretability data that supports\ninteractive user interfaces with hover explanations and click-to-zoom functionality.",
        "operationId": "get_interactive_interpretability_api_interpretability__analysis_id__interactive_get",
        "parameters": [
          {
            "name": "analysis_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Analysis Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/InteractiveInterpretabilityResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/interpretability/{analysis_id}/simplified": {
      "get": {
        "tags": [
          "interpretability"
        ],
        "summary": "Get Simplified Explanation",
        "description": "Get simplified, non-technical explanations suitable for educators and parents.\n\nThis endpoint provides explanations adapted for different user roles\nwith accessible language and clear recommendations.",
        "operationId": "get_simplified_explanation_api_interpretability__analysis_id__simplified_get",
        "parameters": [
          {
            "name": "analysis_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Analysis Id"
            }
          },
          {
            "name": "user_role",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ],
              "default": "educator",
              "title": "User Role"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SimplifiedExplanationResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/interpretability/{analysis_id}/confidence": {
      "get": {
        "tags": [
          "interpretability"
        ],
        "summary": "Get Confidence Metrics",
        "description": "Get confidence metrics and reliability scores for interpretability results.\n\nThis endpoint provides detailed confidence information to help users\nassess the trustworthiness of the analysis and interpretations.",
        "operationId": "get_confidence_metrics_api_interpretability__analysis_id__confidence_get",
        "parameters": [
          {
            "name": "analysis_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Analysis Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ConfidenceMetricsResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/interpretability/{analysis_id}/export": {
      "post": {
        "tags": [
          "interpretability"
        ],
        "summary": "Export Interpretability Results",
        "description": "Export interpretability results in multiple formats (PDF, PNG, CSV, JSON, HTML).\n\nThis endpoint allows users to export comprehensive interpretability reports\nwith customizable options for different use cases.",
        "operationId": "export_interpretability_results_api_interpretability__analysis_id__export_post",
        "parameters": [
          {
            "name": "analysis_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Analysis Id"
            }
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ExportRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ExportResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/interpretability/examples": {
      "get": {
        "tags": [
          "interpretability"
        ],
        "summary": "Get Example Patterns",
        "description": "Get example interpretation patterns for educational purposes.\n\nThis endpoint provides a gallery of common interpretation patterns\nwith explanations suitable for different user roles.",
        "operationId": "get_example_patterns_api_interpretability_examples_get",
        "parameters": [
          {
            "name": "age_group",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ],
              "title": "Age Group"
            }
          },
          {
            "name": "user_role",
            "in": "query",
            "required": false,
            "schema": {
              "type": "string",
              "default": "educator",
              "title": "User Role"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "type": "array",
                  "items": {
                    "type": "object",
                    "additionalProperties": true
                  },
                  "title": "Response Get Example Patterns Api Interpretability Examples Get"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/interpretability/examples/{age_group}": {
      "get": {
        "tags": [
          "interpretability"
        ],
        "summary": "Get Comparison Examples",
        "description": "Get comparison examples for educational purposes from a specific age group.\n\nThis endpoint provides examples of normal and anomalous drawings\nto help users understand typical patterns and variations. Now supports\nfiltering by subject category for more targeted comparisons.",
        "operationId": "get_comparison_examples_api_interpretability_examples__age_group__get",
        "parameters": [
          {
            "name": "age_group",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Age Group"
            }
          },
          {
            "name": "example_type",
            "in": "query",
            "required": false,
            "schema": {
              "type": "string",
              "default": "both",
              "title": "Example Type"
            }
          },
          {
            "name": "subject",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ],
              "title": "Subject"
            }
          },
          {
            "name": "limit",
            "in": "query",
            "required": false,
            "schema": {
              "type": "integer",
              "default": 5,
              "title": "Limit"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ComparisonExamplesResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/interpretability/{analysis_id}/attribution": {
      "get": {
        "tags": [
          "interpretability"
        ],
        "summary": "Get Anomaly Attribution",
        "description": "Get detailed anomaly attribution breakdown (age vs subject vs visual).\n\nThis endpoint provides detailed information about what contributed\nto the anomaly detection: age-related factors, subject-specific factors,\nor visual characteristics.",
        "operationId": "get_anomaly_attribution_api_interpretability__analysis_id__attribution_get",
        "parameters": [
          {
            "name": "analysis_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Analysis Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/interpretability/{analysis_id}/annotate": {
      "post": {
        "tags": [
          "interpretability"
        ],
        "summary": "Add Annotation",
        "description": "Add user annotations to interpretability results.\n\nThis endpoint allows users to add their own notes and observations\nto interpretability results for future reference.",
        "operationId": "add_annotation_api_interpretability__analysis_id__annotate_post",
        "parameters": [
          {
            "name": "analysis_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Analysis Id"
            }
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/AnnotationRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/models/age-groups": {
      "get": {
        "tags": [
          "models"
        ],
        "summary": "List Age Group Models",
        "description": "List available age group models.\n\nThis endpoint returns all age group models with their status,\nsample counts, and threshold information.",
        "operationId": "list_age_group_models_api_models_age_groups_get",
        "parameters": [
          {
            "name": "active_only",
            "in": "query",
            "required": false,
            "schema": {
              "type": "boolean",
              "default": true,
              "title": "Active Only"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ModelListResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/models/train": {
      "post": {
        "tags": [
          "models"
        ],
        "summary": "Train Age Group Model",
        "description": "Train new age group model.\n\nThis endpoint starts training a new autoencoder model for the specified\nage range. Training is performed in the background and progress can be\ntracked using the returned job ID.",
        "operationId": "train_age_group_model_api_models_train_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ModelTrainingRequest"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/models/training/{job_id}/status": {
      "get": {
        "tags": [
          "models"
        ],
        "summary": "Get Training Status",
        "description": "Get training job status.",
        "operationId": "get_training_status_api_models_training__job_id__status_get",
        "parameters": [
          {
            "name": "job_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Job Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/models/{model_id}/threshold": {
      "put": {
        "tags": [
          "models"
        ],
        "summary": "Update Model Threshold",
        "description": "Update model threshold.\n\nThis endpoint allows updating the anomaly detection threshold\nfor a specific age group model. The threshold can be set directly\nor calculated from a percentile of validation data.",
        "operationId": "update_model_threshold_api_models__model_id__threshold_put",
        "parameters": [
          {
            "name": "model_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Model Id"
            }
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ThresholdUpdateRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "additionalProperties": true,
                  "title": "Response Update Model Threshold Api Models  Model Id  Threshold Put"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/models/status": {
      "get": {
        "tags": [
          "models"
        ],
        "summary": "Get Model Status",
        "description": "Get model training and system status.\n\nThis endpoint provides an overview of the model management system,\nincluding counts of models in different states and overall system health.",
        "operationId": "get_model_status_api_models_status_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ModelStatusResponse"
                }
              }
            }
          }
        }
      }
    },
    "/api/models/auto-create": {
      "post": {
        "tags": [
          "models"
        ],
        "summary": "Auto Create Age Groups",
        "description": "Automatically create age group models based on data distribution.\n\nThis endpoint analyzes the available drawing data and creates\nappropriate age group models with sufficient sample sizes.",
        "operationId": "auto_create_age_groups_api_models_auto_create_post",
        "parameters": [
          {
            "name": "force_recreate",
            "in": "query",
            "required": false,
            "schema": {
              "type": "boolean",
              "default": false,
              "title": "Force Recreate"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/models/creation/{job_id}/status": {
      "get": {
        "tags": [
          "models"
        ],
        "summary": "Get Creation Status",
        "description": "Get model creation job status.",
        "operationId": "get_creation_status_api_models_creation__job_id__status_get",
        "parameters": [
          {
            "name": "job_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Job Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/models/{model_id}": {
      "delete": {
        "tags": [
          "models"
        ],
        "summary": "Delete Model",
        "description": "Delete (deactivate) an age group model.\n\nThis endpoint deactivates a model rather than permanently deleting it\nto preserve analysis history.",
        "operationId": "delete_model_api_models__model_id__delete",
        "parameters": [
          {
            "name": "model_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Model Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/models/data-sufficiency/analyze": {
      "get": {
        "tags": [
          "models"
        ],
        "summary": "Analyze Data Sufficiency",
        "description": "Analyze data sufficiency for age groups.\n\nThis endpoint analyzes the available data for specified age groups\nand provides warnings about insufficient data, unbalanced distributions,\nand other data quality issues.\n\nArgs:\n    age_groups: Comma-separated list of age ranges (e.g., \"3-4,4-5,5-6\")\n               If not provided, analyzes all existing age group models",
        "operationId": "analyze_data_sufficiency_api_models_data_sufficiency_analyze_get",
        "parameters": [
          {
            "name": "age_groups",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ],
              "title": "Age Groups"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/models/data-sufficiency/age-group/{age_min}/{age_max}": {
      "get": {
        "tags": [
          "models"
        ],
        "summary": "Analyze Specific Age Group",
        "description": "Analyze data sufficiency for a specific age group.\n\nThis endpoint provides detailed analysis of data availability,\nquality, and distribution for a single age group.",
        "operationId": "analyze_specific_age_group_api_models_data_sufficiency_age_group__age_min___age_max__get",
        "parameters": [
          {
            "name": "age_min",
            "in": "path",
            "required": true,
            "schema": {
              "type": "number",
              "title": "Age Min"
            }
          },
          {
            "name": "age_max",
            "in": "path",
            "required": true,
            "schema": {
              "type": "number",
              "title": "Age Max"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/models/data-sufficiency/merge-age-groups": {
      "post": {
        "tags": [
          "models"
        ],
        "summary": "Merge Age Groups",
        "description": "Merge age groups to improve data sufficiency.\n\nThis endpoint deactivates the original age group models and creates\na new merged age group model with combined data.",
        "operationId": "merge_age_groups_api_models_data_sufficiency_merge_age_groups_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/Body_merge_age_groups_api_models_data_sufficiency_merge_age_groups_post"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/models/data-sufficiency/warnings": {
      "get": {
        "tags": [
          "models"
        ],
        "summary": "Get Data Warnings",
        "description": "Get data sufficiency warnings for all age groups.\n\nThis endpoint returns warnings about data quality issues,\noptionally filtered by severity level.",
        "operationId": "get_data_warnings_api_models_data_sufficiency_warnings_get",
        "parameters": [
          {
            "name": "severity",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ],
              "title": "Severity"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/training/jobs": {
      "post": {
        "tags": [
          "training"
        ],
        "summary": "Submit Training Job",
        "description": "Submit a new training job to either local or SageMaker environment.\n\nThis endpoint creates and submits a training job based on the specified\nenvironment. For SageMaker jobs, it handles container building, data upload,\nand job submission. For local jobs, it starts training immediately.",
        "operationId": "submit_training_job_api_training_jobs_post",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/TrainingConfigRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/TrainingJobResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      },
      "get": {
        "tags": [
          "training"
        ],
        "summary": "List Training Jobs",
        "description": "List training jobs with optional filtering.\n\nThis endpoint returns a list of training jobs, optionally filtered\nby environment (local/sagemaker) and status.",
        "operationId": "list_training_jobs_api_training_jobs_get",
        "parameters": [
          {
            "name": "environment",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "$ref": "#/components/schemas/TrainingEnvironment"
                },
                {
                  "type": "null"
                }
              ],
              "title": "Environment"
            }
          },
          {
            "name": "status",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ],
              "title": "Status"
            }
          },
          {
            "name": "limit",
            "in": "query",
            "required": false,
            "schema": {
              "type": "integer",
              "default": 50,
              "title": "Limit"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "type": "array",
                  "items": {
                    "$ref": "#/components/schemas/TrainingJobResponse"
                  },
                  "title": "Response List Training Jobs Api Training Jobs Get"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/training/jobs/{job_id}": {
      "get": {
        "tags": [
          "training"
        ],
        "summary": "Get Training Job Status",
        "description": "Get detailed status of a specific training job.\n\nThis endpoint returns comprehensive information about a training job,\nincluding progress, metrics, and environment-specific details.",
        "operationId": "get_training_job_status_api_training_jobs__job_id__get",
        "parameters": [
          {
            "name": "job_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Job Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "additionalProperties": true,
                  "title": "Response Get Training Job Status Api Training Jobs  Job Id  Get"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/training/jobs/{job_id}/cancel": {
      "post": {
        "tags": [
          "training"
        ],
        "summary": "Cancel Training Job",
        "description": "Cancel a running training job.\n\nThis endpoint attempts to cancel a training job. For local jobs,\nit stops the training process. For SageMaker jobs, it stops the\nSageMaker training job.",
        "operationId": "cancel_training_job_api_training_jobs__job_id__cancel_post",
        "parameters": [
          {
            "name": "job_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Job Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/training/jobs/{job_id}/reports": {
      "get": {
        "tags": [
          "training"
        ],
        "summary": "Get Training Reports",
        "description": "Get training reports for a specific job.\n\nThis endpoint returns all training reports associated with a job,\nincluding metrics, model paths, and performance summaries.",
        "operationId": "get_training_reports_api_training_jobs__job_id__reports_get",
        "parameters": [
          {
            "name": "job_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Job Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "type": "array",
                  "items": {
                    "$ref": "#/components/schemas/TrainingReportResponse"
                  },
                  "title": "Response Get Training Reports Api Training Jobs  Job Id  Reports Get"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/training/deploy": {
      "post": {
        "tags": [
          "training"
        ],
        "summary": "Deploy Trained Model",
        "description": "Deploy trained model parameters to production system.\n\nThis endpoint loads trained model parameters and creates a new\nage group model for production use.",
        "operationId": "deploy_trained_model_api_training_deploy_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ModelDeploymentRequest"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/training/environments/status": {
      "get": {
        "tags": [
          "training"
        ],
        "summary": "Get Training Environments Status",
        "description": "Get status of available training environments.\n\nThis endpoint returns information about local and SageMaker\ntraining environments, including availability and configuration.",
        "operationId": "get_training_environments_status_api_training_environments_status_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/training/sagemaker/setup": {
      "post": {
        "tags": [
          "training"
        ],
        "summary": "Setup Sagemaker Environment",
        "description": "Setup SageMaker training environment.\n\nThis endpoint helps set up the necessary AWS resources for\nSageMaker training, including IAM roles and container repositories.",
        "operationId": "setup_sagemaker_environment_api_training_sagemaker_setup_post",
        "parameters": [
          {
            "name": "s3_bucket",
            "in": "query",
            "required": true,
            "schema": {
              "type": "string",
              "title": "S3 Bucket"
            }
          },
          {
            "name": "ecr_repository",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ],
              "title": "Ecr Repository"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/training/models/export": {
      "post": {
        "tags": [
          "training"
        ],
        "summary": "Export Model From Training Job",
        "description": "Export trained model from training job in production-compatible format.\n\nThis endpoint exports a trained model from a completed training job,\ncreating a production-ready model file with metadata and validation.",
        "operationId": "export_model_from_training_job_api_training_models_export_post",
        "parameters": [
          {
            "name": "training_job_id",
            "in": "query",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Training Job Id"
            }
          },
          {
            "name": "age_group_min",
            "in": "query",
            "required": true,
            "schema": {
              "type": "number",
              "title": "Age Group Min"
            }
          },
          {
            "name": "age_group_max",
            "in": "query",
            "required": true,
            "schema": {
              "type": "number",
              "title": "Age Group Max"
            }
          },
          {
            "name": "export_format",
            "in": "query",
            "required": false,
            "schema": {
              "type": "string",
              "default": "pytorch",
              "title": "Export Format"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/training/models/exports": {
      "get": {
        "tags": [
          "training"
        ],
        "summary": "List Exported Models",
        "description": "List all exported models with their metadata.\n\nThis endpoint returns a list of all models that have been exported,\nincluding their metadata, export timestamps, and file information.",
        "operationId": "list_exported_models_api_training_models_exports_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/training/models/validate": {
      "post": {
        "tags": [
          "training"
        ],
        "summary": "Validate Exported Model",
        "description": "Validate exported model for compatibility and integrity.\n\nThis endpoint performs comprehensive validation of an exported model,\nchecking file integrity, compatibility, and performance metrics.",
        "operationId": "validate_exported_model_api_training_models_validate_post",
        "parameters": [
          {
            "name": "model_id",
            "in": "query",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Model Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/training/models/deploy": {
      "post": {
        "tags": [
          "training"
        ],
        "summary": "Deploy Exported Model",
        "description": "Deploy exported model to production environment.\n\nThis endpoint deploys an exported model to the production system,\nmaking it available for anomaly detection in the specified age group.",
        "operationId": "deploy_exported_model_api_training_models_deploy_post",
        "parameters": [
          {
            "name": "model_export_path",
            "in": "query",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Model Export Path"
            }
          },
          {
            "name": "age_group_min",
            "in": "query",
            "required": true,
            "schema": {
              "type": "number",
              "title": "Age Group Min"
            }
          },
          {
            "name": "age_group_max",
            "in": "query",
            "required": true,
            "schema": {
              "type": "number",
              "title": "Age Group Max"
            }
          },
          {
            "name": "replace_existing",
            "in": "query",
            "required": false,
            "schema": {
              "type": "boolean",
              "default": false,
              "title": "Replace Existing"
            }
          },
          {
            "name": "validate_before_deployment",
            "in": "query",
            "required": false,
            "schema": {
              "type": "boolean",
              "default": true,
              "title": "Validate Before Deployment"
            }
          },
          {
            "name": "backup_existing",
            "in": "query",
            "required": false,
            "schema": {
              "type": "boolean",
              "default": true,
              "title": "Backup Existing"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/training/models/deployed": {
      "get": {
        "tags": [
          "training"
        ],
        "summary": "List Deployed Models",
        "description": "List all deployed models in production.\n\nThis endpoint returns information about all models currently\ndeployed and active in the production system.",
        "operationId": "list_deployed_models_api_training_models_deployed_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/training/models/{model_id}/undeploy": {
      "post": {
        "tags": [
          "training"
        ],
        "summary": "Undeploy Model",
        "description": "Undeploy (deactivate) a deployed model.\n\nThis endpoint deactivates a deployed model, removing it from\nactive use in the production system.",
        "operationId": "undeploy_model_api_training_models__model_id__undeploy_post",
        "parameters": [
          {
            "name": "model_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Model Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/config/": {
      "get": {
        "tags": [
          "configuration"
        ],
        "summary": "Get Config",
        "description": "Get current system configuration.\n\nThis endpoint returns the current system configuration including\nmodel settings, threshold parameters, and age grouping strategy.",
        "operationId": "get_config_api_config__get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SystemConfigurationResponse"
                }
              }
            }
          }
        }
      },
      "put": {
        "tags": [
          "configuration"
        ],
        "summary": "Update Config",
        "description": "Update system configuration.\n\nThis endpoint updates various system configuration settings\nincluding thresholds and age grouping parameters.",
        "operationId": "update_config_api_config__put",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ConfigurationUpdateRequest"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SuccessResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/config/threshold": {
      "put": {
        "tags": [
          "configuration"
        ],
        "summary": "Update Threshold Settings",
        "description": "Update global threshold settings.\n\nThis endpoint recalculates thresholds for all active models\nusing the specified percentile value from the request body.",
        "operationId": "update_threshold_settings_api_config_threshold_put",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ThresholdUpdateRequest"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SuccessResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/config/age-grouping": {
      "put": {
        "tags": [
          "configuration"
        ],
        "summary": "Update Age Grouping",
        "description": "Modify age grouping strategy.\n\nThis endpoint updates the age grouping configuration and can\noptionally trigger recreation of age group models.",
        "operationId": "update_age_grouping_api_config_age_grouping_put",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ConfigurationUpdateRequest"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SuccessResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/config/health": {
      "get": {
        "tags": [
          "configuration"
        ],
        "summary": "Health Check",
        "description": "System health check endpoint.\n\nThis endpoint provides information about the health and status\nof various system components.",
        "operationId": "health_check_api_config_health_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HealthCheckResponse"
                }
              }
            }
          }
        }
      }
    },
    "/api/config/stats": {
      "get": {
        "tags": [
          "configuration"
        ],
        "summary": "Get System Stats",
        "description": "Get comprehensive system statistics.\n\nThis endpoint provides detailed statistics about the system\nincluding data distribution, model performance, and usage metrics.",
        "operationId": "get_system_stats_api_config_stats_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/config/subjects": {
      "get": {
        "tags": [
          "configuration"
        ],
        "summary": "Get Supported Subject Categories",
        "description": "Get list of supported subject categories.\n\nThis endpoint returns all supported subject categories that can be used\nwhen uploading drawings, along with usage statistics.",
        "operationId": "get_supported_subject_categories_api_config_subjects_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/config/subjects/statistics": {
      "get": {
        "tags": [
          "configuration"
        ],
        "summary": "Get Subject Specific Statistics",
        "description": "Get subject-specific statistics and analysis data.\n\nThis endpoint provides detailed statistics about drawings and analyses\nfor specific subject categories or overall subject-related metrics.",
        "operationId": "get_subject_specific_statistics_api_config_subjects_statistics_get",
        "parameters": [
          {
            "name": "subject",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ],
              "title": "Subject"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/config/models/subject-aware": {
      "get": {
        "tags": [
          "configuration"
        ],
        "summary": "Get Subject Aware Model Status",
        "description": "Get status of subject-aware model capabilities.\n\nThis endpoint provides information about the current subject-aware\nmodeling capabilities and model status.",
        "operationId": "get_subject_aware_model_status_api_config_models_subject_aware_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/config/reset": {
      "post": {
        "tags": [
          "configuration"
        ],
        "summary": "Reset System",
        "description": "Reset system configuration and models.\n\nWARNING: This endpoint deactivates all models and clears caches.\nUse with caution in production environments.",
        "operationId": "reset_system_api_config_reset_post",
        "parameters": [
          {
            "name": "confirm",
            "in": "query",
            "required": false,
            "schema": {
              "type": "boolean",
              "default": false,
              "title": "Confirm"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/documentation/status": {
      "get": {
        "tags": [
          "documentation"
        ],
        "summary": "Get Documentation Status",
        "description": "Get current documentation generation status.\n\nReturns real-time status of documentation generation including progress,\ncurrent task, and any errors or warnings.",
        "operationId": "get_documentation_status_api_documentation_status_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/DocumentationStatus"
                }
              }
            }
          }
        }
      }
    },
    "/api/documentation/metrics": {
      "get": {
        "tags": [
          "documentation"
        ],
        "summary": "Get Documentation Metrics",
        "description": "Get comprehensive documentation metrics.\n\nReturns metrics about documentation files, generation history,\nsuccess rates, and validation status.",
        "operationId": "get_documentation_metrics_api_documentation_metrics_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/DocumentationMetrics"
                }
              }
            }
          }
        }
      }
    },
    "/api/documentation/generate": {
      "post": {
        "tags": [
          "documentation"
        ],
        "summary": "Generate Documentation",
        "description": "Trigger documentation generation.\n\nStarts documentation generation process in the background.\nUse the status endpoint to monitor progress.",
        "operationId": "generate_documentation_api_documentation_generate_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/GenerationRequest"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/GenerationResult"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/documentation/generate/sync": {
      "post": {
        "tags": [
          "documentation"
        ],
        "summary": "Generate Documentation Sync",
        "description": "Generate documentation synchronously.\n\nRuns documentation generation and waits for completion.\nUse this for smaller generation tasks or when immediate results are needed.",
        "operationId": "generate_documentation_sync_api_documentation_generate_sync_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/GenerationRequest"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/GenerationResult"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/documentation/categories": {
      "get": {
        "tags": [
          "documentation"
        ],
        "summary": "Get Documentation Categories",
        "description": "Get available documentation categories.\n\nReturns list of available documentation categories that can be generated.",
        "operationId": "get_documentation_categories_api_documentation_categories_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/documentation/files": {
      "get": {
        "tags": [
          "documentation"
        ],
        "summary": "Get Documentation Files",
        "description": "Get list of documentation files with metadata.\n\nReturns comprehensive list of documentation files with metadata,\nfiltering, and search capabilities.",
        "operationId": "get_documentation_files_api_documentation_files_get",
        "parameters": [
          {
            "name": "category",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ],
              "description": "Filter by category",
              "title": "Category"
            },
            "description": "Filter by category"
          },
          {
            "name": "search",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ],
              "description": "Search in file names and content",
              "title": "Search"
            },
            "description": "Search in file names and content"
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/documentation/cache": {
      "delete": {
        "tags": [
          "documentation"
        ],
        "summary": "Clear Documentation Cache",
        "description": "Clear documentation generation cache.\n\nForces regeneration of all documentation by clearing the cache.",
        "operationId": "clear_documentation_cache_api_documentation_cache_delete",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/documentation/validation": {
      "get": {
        "tags": [
          "documentation"
        ],
        "summary": "Get Validation Status",
        "description": "Get comprehensive validation status for all documentation.\n\nReturns detailed validation results including errors, warnings,\nand quality metrics.",
        "operationId": "get_validation_status_api_documentation_validation_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/documentation/validate": {
      "post": {
        "tags": [
          "documentation"
        ],
        "summary": "Validate Documentation",
        "description": "Run validation on documentation.\n\nValidates documentation for technical accuracy, link integrity,\naccessibility compliance, and formatting consistency.",
        "operationId": "validate_documentation_api_documentation_validate_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "anyOf": [
                  {
                    "items": {
                      "type": "string"
                    },
                    "type": "array"
                  },
                  {
                    "type": "null"
                  }
                ],
                "title": "Categories"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/documentation/preview/{category}": {
      "get": {
        "tags": [
          "documentation"
        ],
        "summary": "Preview Documentation Changes",
        "description": "Preview documentation changes before generation.\n\nShows what would be generated for a specific category or file\nwithout actually writing the files.",
        "operationId": "preview_documentation_changes_api_documentation_preview__category__get",
        "parameters": [
          {
            "name": "category",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Category"
            }
          },
          {
            "name": "file_path",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ],
              "description": "Specific file to preview",
              "title": "File Path"
            },
            "description": "Specific file to preview"
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/documentation/batch/generate": {
      "post": {
        "tags": [
          "documentation"
        ],
        "summary": "Batch Generate Documentation",
        "description": "Batch generate multiple documentation categories with scheduling.\n\nAllows generating multiple categories in sequence with different\nconfigurations for each category.",
        "operationId": "batch_generate_documentation_api_documentation_batch_generate_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "additionalProperties": true,
                "type": "object",
                "title": "Request"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/documentation/batch/validate": {
      "post": {
        "tags": [
          "documentation"
        ],
        "summary": "Batch Validate Documentation",
        "description": "Batch validate multiple documentation categories.\n\nRuns validation on multiple categories in parallel for faster processing.",
        "operationId": "batch_validate_documentation_api_documentation_batch_validate_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "items": {
                  "type": "string"
                },
                "type": "array",
                "title": "Categories"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/documentation/schedule": {
      "get": {
        "tags": [
          "documentation"
        ],
        "summary": "Get Generation Schedule",
        "description": "Get current generation schedule and queue.\n\nReturns information about scheduled and queued generation tasks.",
        "operationId": "get_generation_schedule_api_documentation_schedule_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      },
      "post": {
        "tags": [
          "documentation"
        ],
        "summary": "Schedule Generation",
        "description": "Schedule documentation generation for later execution.\n\nAllows scheduling generation tasks for specific times or intervals.",
        "operationId": "schedule_generation_api_documentation_schedule_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "additionalProperties": true,
                "type": "object",
                "title": "Request"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/documentation/search": {
      "post": {
        "tags": [
          "documentation"
        ],
        "summary": "Search Documentation",
        "description": "Search documentation with advanced filtering and faceting.\n\nProvides full-text search across all documentation with relevance scoring,\nfaceted filtering, and intelligent suggestions.",
        "operationId": "search_documentation_api_documentation_search_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/SearchRequest"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SearchResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/documentation/search/suggestions": {
      "get": {
        "tags": [
          "documentation"
        ],
        "summary": "Get Search Suggestions",
        "description": "Get search suggestions for autocomplete.\n\nProvides intelligent search suggestions based on indexed content\nand common search patterns.",
        "operationId": "get_search_suggestions_api_documentation_search_suggestions_get",
        "parameters": [
          {
            "name": "query",
            "in": "query",
            "required": true,
            "schema": {
              "type": "string",
              "description": "Partial query for suggestions",
              "title": "Query"
            },
            "description": "Partial query for suggestions"
          },
          {
            "name": "limit",
            "in": "query",
            "required": false,
            "schema": {
              "type": "integer",
              "description": "Maximum number of suggestions",
              "default": 10,
              "title": "Limit"
            },
            "description": "Maximum number of suggestions"
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/documentation/search/statistics": {
      "get": {
        "tags": [
          "documentation"
        ],
        "summary": "Get Search Statistics",
        "description": "Get search index statistics.\n\nReturns comprehensive statistics about the search index including\ndocument counts, index size, and performance metrics.",
        "operationId": "get_search_statistics_api_documentation_search_statistics_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/documentation/search/index": {
      "post": {
        "tags": [
          "documentation"
        ],
        "summary": "Rebuild Search Index",
        "description": "Rebuild the search index.\n\nRebuilds the search index from all documentation files.\nUse force=true to completely rebuild the index.",
        "operationId": "rebuild_search_index_api_documentation_search_index_post",
        "parameters": [
          {
            "name": "force",
            "in": "query",
            "required": false,
            "schema": {
              "type": "boolean",
              "description": "Force complete reindexing",
              "default": false,
              "title": "Force"
            },
            "description": "Force complete reindexing"
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/documentation/navigation/{document_id}": {
      "get": {
        "tags": [
          "documentation"
        ],
        "summary": "Get Navigation Context",
        "description": "Get navigation context for a document.\n\nReturns comprehensive navigation context including breadcrumbs,\ncross-references, related content, and sequential navigation.",
        "operationId": "get_navigation_context_api_documentation_navigation__document_id__get",
        "parameters": [
          {
            "name": "document_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Document Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/documentation/navigation/sitemap": {
      "get": {
        "tags": [
          "documentation"
        ],
        "summary": "Get Sitemap",
        "description": "Get complete documentation sitemap.\n\nReturns hierarchical sitemap of all documentation organized by type\nand category with metadata.",
        "operationId": "get_sitemap_api_documentation_navigation_sitemap_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/documentation/navigation/cross-references": {
      "get": {
        "tags": [
          "documentation"
        ],
        "summary": "Get Cross Reference Report",
        "description": "Get cross-reference analysis report.\n\nReturns comprehensive analysis of cross-references including\nbroken links, most referenced documents, and orphaned content.",
        "operationId": "get_cross_reference_report_api_documentation_navigation_cross_references_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/documentation/navigation/rebuild": {
      "post": {
        "tags": [
          "documentation"
        ],
        "summary": "Rebuild Navigation Structure",
        "description": "Rebuild navigation structure.\n\nRebuilds the navigation structure and cross-reference index\nfrom all documentation files.",
        "operationId": "rebuild_navigation_structure_api_documentation_navigation_rebuild_post",
        "parameters": [
          {
            "name": "force",
            "in": "query",
            "required": false,
            "schema": {
              "type": "boolean",
              "description": "Force complete rebuild",
              "default": false,
              "title": "Force"
            },
            "description": "Force complete rebuild"
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/metrics/usage": {
      "get": {
        "tags": [
          "metrics"
        ],
        "summary": "Get Usage Metrics",
        "description": "Get comprehensive usage metrics for the dashboard.\n\nReturns metrics including:\n- Total analyses and drawings\n- Time-based analysis counts (daily, weekly, monthly)\n- Active user sessions and geographic distribution\n- System health and performance metrics\n- Processing time statistics",
        "operationId": "get_usage_metrics_api_metrics_usage_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "additionalProperties": true,
                  "type": "object",
                  "title": "Response Get Usage Metrics Api Metrics Usage Get"
                }
              }
            }
          }
        }
      }
    },
    "/api/metrics/health": {
      "get": {
        "tags": [
          "metrics"
        ],
        "summary": "Get System Health",
        "description": "Get system health metrics including uptime, error rates, and resource usage.",
        "operationId": "get_system_health_api_metrics_health_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "additionalProperties": true,
                  "type": "object",
                  "title": "Response Get System Health Api Metrics Health Get"
                }
              }
            }
          }
        }
      }
    },
    "/api/metrics/sessions": {
      "get": {
        "tags": [
          "metrics"
        ],
        "summary": "Get Session Metrics",
        "description": "Get current user session metrics and geographic distribution.",
        "operationId": "get_session_metrics_api_metrics_sessions_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "additionalProperties": true,
                  "type": "object",
                  "title": "Response Get Session Metrics Api Metrics Sessions Get"
                }
              }
            }
          }
        }
      }
    },
    "/api/metrics/performance": {
      "get": {
        "tags": [
          "metrics"
        ],
        "summary": "Get Performance Metrics",
        "description": "Get detailed performance metrics including processing times and system resources.",
        "operationId": "get_performance_metrics_api_metrics_performance_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "additionalProperties": true,
                  "type": "object",
                  "title": "Response Get Performance Metrics Api Metrics Performance Get"
                }
              }
            }
          }
        }
      }
    },
    "/api/metrics/session/start": {
      "post": {
        "tags": [
          "metrics"
        ],
        "summary": "Start User Session",
        "description": "Manually start a user session (alternative to automatic detection).\n\nRequest body should contain:\n- ip_address: Client IP address\n- user_agent: User agent string",
        "operationId": "start_user_session_api_metrics_session_start_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "additionalProperties": {
                  "type": "string"
                },
                "type": "object",
                "title": "Request Info"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "additionalProperties": true,
                  "type": "object",
                  "title": "Response Start User Session Api Metrics Session Start Post"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/metrics/session/{session_id}/end": {
      "post": {
        "tags": [
          "metrics"
        ],
        "summary": "End User Session",
        "description": "Manually end a user session.",
        "operationId": "end_user_session_api_metrics_session__session_id__end_post",
        "parameters": [
          {
            "name": "session_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Session Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "additionalProperties": true,
                  "title": "Response End User Session Api Metrics Session  Session Id  End Post"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/metrics/cloudwatch/status": {
      "get": {
        "tags": [
          "metrics"
        ],
        "summary": "Get Cloudwatch Status",
        "description": "Get CloudWatch integration status and configuration.",
        "operationId": "get_cloudwatch_status_api_metrics_cloudwatch_status_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "additionalProperties": true,
                  "type": "object",
                  "title": "Response Get Cloudwatch Status Api Metrics Cloudwatch Status Get"
                }
              }
            }
          }
        }
      }
    },
    "/api/demo/": {
      "get": {
        "tags": [
          "demo"
        ],
        "summary": "Get Demo Page",
        "description": "Get the complete demo page with all content.\n\nReturns:\n    HTML response with complete demo page content",
        "operationId": "get_demo_page_api_demo__get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "text/html": {
                "schema": {
                  "type": "string"
                }
              }
            }
          }
        }
      }
    },
    "/api/demo/samples": {
      "get": {
        "tags": [
          "demo"
        ],
        "summary": "Get Demo Samples",
        "description": "Get all demo samples with analysis results.\n\nReturns:\n    List of demo samples with complete analysis data",
        "operationId": "get_demo_samples_api_demo_samples_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SuccessResponse"
                }
              }
            }
          }
        }
      }
    },
    "/api/demo/samples/{sample_id}": {
      "get": {
        "tags": [
          "demo"
        ],
        "summary": "Get Demo Sample",
        "description": "Get a specific demo sample by ID.\n\nArgs:\n    sample_id: ID of the demo sample\n\nReturns:\n    Demo sample with complete analysis data",
        "operationId": "get_demo_sample_api_demo_samples__sample_id__get",
        "parameters": [
          {
            "name": "sample_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Sample Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SuccessResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/demo/project-info": {
      "get": {
        "tags": [
          "demo"
        ],
        "summary": "Get Project Info",
        "description": "Get comprehensive project information for demo page.\n\nReturns:\n    Project description with technical details and features",
        "operationId": "get_project_info_api_demo_project_info_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SuccessResponse"
                }
              }
            }
          }
        }
      }
    },
    "/api/demo/disclaimer": {
      "get": {
        "tags": [
          "demo"
        ],
        "summary": "Get Medical Disclaimer",
        "description": "Get medical disclaimer and warnings for demo content.\n\nReturns:\n    Medical disclaimer with all required warnings",
        "operationId": "get_medical_disclaimer_api_demo_disclaimer_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SuccessResponse"
                }
              }
            }
          }
        }
      }
    },
    "/api/demo/technical-links": {
      "get": {
        "tags": [
          "demo"
        ],
        "summary": "Get Technical Links",
        "description": "Get technical links and documentation references.\n\nReturns:\n    Technical links including GitHub repository and documentation",
        "operationId": "get_technical_links_api_demo_technical_links_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SuccessResponse"
                }
              }
            }
          }
        }
      }
    },
    "/api/demo/statistics": {
      "get": {
        "tags": [
          "demo"
        ],
        "summary": "Get Demo Statistics",
        "description": "Get demo-specific statistics and metrics.\n\nReturns:\n    Demo statistics including sample counts and distributions",
        "operationId": "get_demo_statistics_api_demo_statistics_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SuccessResponse"
                }
              }
            }
          }
        }
      }
    },
    "/api/files/s3/{file_path}": {
      "get": {
        "tags": [
          "files"
        ],
        "summary": "Serve S3 File",
        "description": "Serve a file from S3 storage.\n\nThis endpoint downloads files from S3 and serves them with proper caching headers.\nThis avoids presigned URL expiration issues and allows CloudFront to cache responses.\n\nArgs:\n    file_path: S3 key path (e.g., \"drawings/20240108_123456_abc123.png\")\n\nReturns:\n    File response with caching headers",
        "operationId": "serve_s3_file_api_files_s3__file_path__get",
        "parameters": [
          {
            "name": "file_path",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "File Path"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      },
      "head": {
        "tags": [
          "files"
        ],
        "summary": "Check S3 File",
        "description": "Check if a file exists in S3 storage.\n\nArgs:\n    file_path: S3 key path\n\nReturns:\n    Empty response with appropriate status code",
        "operationId": "check_s3_file_api_files_s3__file_path__head",
        "parameters": [
          {
            "name": "file_path",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "File Path"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/files/markdown": {
      "get": {
        "tags": [
          "files"
        ],
        "summary": "Serve Markdown File",
        "description": "Serve a markdown file from the local filesystem.\n\nArgs:\n    path: Relative path to markdown file (e.g., \"tmp_files/analysis.md\")\n\nReturns:\n    Markdown file content as plain text",
        "operationId": "serve_markdown_file_api_files_markdown_get",
        "parameters": [
          {
            "name": "path",
            "in": "query",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Path"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/database/backup": {
      "post": {
        "tags": [
          "database"
        ],
        "summary": "Create Database Backup",
        "description": "Create a database backup with optional S3 upload.\n\n- **upload_to_s3**: Whether to upload to S3 (defaults to environment setting)\n- **include_files**: Whether to include uploaded files and static content",
        "operationId": "create_database_backup_api_database_backup_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/BackupRequest"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "additionalProperties": true,
                  "type": "object",
                  "title": "Response Create Database Backup Api Database Backup Post"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/database/migrate": {
      "post": {
        "tags": [
          "database"
        ],
        "summary": "Run Database Migration",
        "description": "Run database migrations to the specified revision.\n\n- **target_revision**: Target migration revision (defaults to \"head\")",
        "operationId": "run_database_migration_api_database_migrate_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/MigrationRequest"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "additionalProperties": true,
                  "type": "object",
                  "title": "Response Run Database Migration Api Database Migrate Post"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/database/migration-info": {
      "get": {
        "tags": [
          "database"
        ],
        "summary": "Get Migration Info",
        "description": "Get current database migration information.",
        "operationId": "get_migration_info_api_database_migration_info_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "additionalProperties": true,
                  "type": "object",
                  "title": "Response Get Migration Info Api Database Migration Info Get"
                }
              }
            }
          }
        }
      }
    },
    "/api/database/validate-consistency": {
      "post": {
        "tags": [
          "database"
        ],
        "summary": "Validate Cross Environment Consistency",
        "description": "Validate database schema consistency across environments.\n\n- **other_db_url**: Database URL of the other environment to compare",
        "operationId": "validate_cross_environment_consistency_api_database_validate_consistency_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ConsistencyCheckRequest"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "additionalProperties": true,
                  "type": "object",
                  "title": "Response Validate Cross Environment Consistency Api Database Validate Consistency Post"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/database/backup-list": {
      "get": {
        "tags": [
          "database"
        ],
        "summary": "List Backups",
        "description": "Get list of available database backups.",
        "operationId": "list_backups_api_database_backup_list_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "additionalProperties": true,
                  "type": "object",
                  "title": "Response List Backups Api Database Backup List Get"
                }
              }
            }
          }
        }
      }
    },
    "/api/database/schedule-backups": {
      "post": {
        "tags": [
          "database"
        ],
        "summary": "Schedule Automated Backups",
        "description": "Schedule automated database backups.\n\n- **interval_hours**: Backup interval in hours (default: 6)",
        "operationId": "schedule_automated_backups_api_database_schedule_backups_post",
        "parameters": [
          {
            "name": "interval_hours",
            "in": "query",
            "required": false,
            "schema": {
              "type": "integer",
              "default": 6,
              "title": "Interval Hours"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "additionalProperties": true,
                  "title": "Response Schedule Automated Backups Api Database Schedule Backups Post"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/database/consistency-check": {
      "post": {
        "tags": [
          "database"
        ],
        "summary": "Run Consistency Check",
        "description": "Run database consistency validation.",
        "operationId": "run_consistency_check_api_database_consistency_check_post",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "additionalProperties": true,
                  "type": "object",
                  "title": "Response Run Consistency Check Api Database Consistency Check Post"
                }
              }
            }
          }
        }
      }
    },
    "/api/security/status": {
      "get": {
        "tags": [
          "security"
        ],
        "summary": "Get Security Status",
        "description": "Get current security service status and configuration.\n\nReturns information about security service initialization,\nAWS client availability, and current security policy.",
        "operationId": "get_security_status_api_security_status_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "additionalProperties": true,
                  "type": "object",
                  "title": "Response Get Security Status Api Security Status Get"
                }
              }
            }
          }
        }
      }
    },
    "/api/security/validate/iam-role": {
      "post": {
        "tags": [
          "security"
        ],
        "summary": "Validate Iam Role",
        "description": "Validate IAM role for least-privilege compliance.\n\nChecks the specified IAM role for overly broad permissions,\ndangerous policy attachments, and compliance with security best practices.",
        "operationId": "validate_iam_role_api_security_validate_iam_role_post",
        "parameters": [
          {
            "name": "role_arn",
            "in": "query",
            "required": true,
            "schema": {
              "type": "string",
              "description": "IAM role ARN to validate",
              "title": "Role Arn"
            },
            "description": "IAM role ARN to validate"
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SecurityValidationResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/security/validate/s3-bucket": {
      "post": {
        "tags": [
          "security"
        ],
        "summary": "Validate S3 Bucket",
        "description": "Validate S3 bucket encryption and security configuration.\n\nChecks the specified S3 bucket for proper encryption configuration,\npublic access blocks, and security compliance.",
        "operationId": "validate_s3_bucket_api_security_validate_s3_bucket_post",
        "parameters": [
          {
            "name": "bucket_name",
            "in": "query",
            "required": true,
            "schema": {
              "type": "string",
              "description": "S3 bucket name to validate",
              "title": "Bucket Name"
            },
            "description": "S3 bucket name to validate"
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SecurityValidationResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/security/validate/security-groups": {
      "post": {
        "tags": [
          "security"
        ],
        "summary": "Validate Security Groups",
        "description": "Validate security group configurations for minimal exposure.\n\nChecks the specified security groups for overly permissive rules,\nopen ports, and compliance with network security best practices.",
        "operationId": "validate_security_groups_api_security_validate_security_groups_post",
        "parameters": [
          {
            "name": "security_group_ids",
            "in": "query",
            "required": true,
            "schema": {
              "type": "array",
              "items": {
                "type": "string"
              },
              "description": "Security group IDs to validate",
              "title": "Security Group Ids"
            },
            "description": "Security group IDs to validate"
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SecurityValidationResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/security/validate/vpc": {
      "post": {
        "tags": [
          "security"
        ],
        "summary": "Validate Vpc",
        "description": "Validate VPC configuration for security compliance.\n\nChecks the specified VPC for proper subnet configuration,\nprivate subnet isolation, and network security best practices.",
        "operationId": "validate_vpc_api_security_validate_vpc_post",
        "parameters": [
          {
            "name": "vpc_id",
            "in": "query",
            "required": true,
            "schema": {
              "type": "string",
              "description": "VPC ID to validate",
              "title": "Vpc Id"
            },
            "description": "VPC ID to validate"
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SecurityValidationResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/security/validate/encryption-in-transit": {
      "get": {
        "tags": [
          "security"
        ],
        "summary": "Validate Encryption In Transit",
        "description": "Validate encryption in transit configuration.\n\nChecks the current deployment for proper HTTPS enforcement,\nTLS configuration, and secure communication protocols.",
        "operationId": "validate_encryption_in_transit_api_security_validate_encryption_in_transit_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SecurityValidationResponse"
                }
              }
            }
          }
        }
      }
    },
    "/api/security/audit/comprehensive": {
      "post": {
        "tags": [
          "security"
        ],
        "summary": "Comprehensive Security Audit",
        "description": "Perform comprehensive security audit of AWS resources.\n\nValidates all specified AWS resources for security compliance,\nincluding IAM roles, S3 buckets, security groups, VPC configuration,\nand encryption settings.",
        "operationId": "comprehensive_security_audit_api_security_audit_comprehensive_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/SecurityAuditRequest"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SecurityAuditResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/security/compliance/report": {
      "get": {
        "tags": [
          "security"
        ],
        "summary": "Get Compliance Report",
        "description": "Generate security compliance report.\n\nCreates a detailed compliance report based on current security\nconfiguration and validation results.",
        "operationId": "get_compliance_report_api_security_compliance_report_get",
        "parameters": [
          {
            "name": "format",
            "in": "query",
            "required": false,
            "schema": {
              "type": "string",
              "description": "Report format: json, csv, or html",
              "default": "json",
              "title": "Format"
            },
            "description": "Report format: json, csv, or html"
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/cost-optimization/estimate": {
      "get": {
        "tags": [
          "cost-optimization"
        ],
        "summary": "Get Cost Estimate",
        "description": "Get estimated monthly costs for optimized AWS resources.\n\nReturns cost breakdown and compliance status for the production deployment.",
        "operationId": "get_cost_estimate_api_cost_optimization_estimate_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/CostEstimateResponse"
                }
              }
            }
          }
        }
      }
    },
    "/api/cost-optimization/optimization": {
      "get": {
        "tags": [
          "cost-optimization"
        ],
        "summary": "Get Cost Optimization",
        "description": "Get cost optimization configurations and recommendations.\n\nReturns optimized configurations for ECS Fargate, S3, and CloudFront.",
        "operationId": "get_cost_optimization_api_cost_optimization_optimization_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/CostOptimizationResponse"
                }
              }
            }
          }
        }
      }
    },
    "/api/cost-optimization/compliance": {
      "get": {
        "tags": [
          "cost-optimization"
        ],
        "summary": "Validate Cost Compliance",
        "description": "Validate cost compliance against budget requirements.\n\nReturns compliance status and detailed cost analysis.",
        "operationId": "validate_cost_compliance_api_cost_optimization_compliance_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/CostComplianceResponse"
                }
              }
            }
          }
        }
      }
    },
    "/api/cost-optimization/apply-s3-lifecycle/{bucket_name}": {
      "post": {
        "tags": [
          "cost-optimization"
        ],
        "summary": "Apply S3 Lifecycle Optimization",
        "description": "Apply S3 lifecycle optimization to a specific bucket.\n\nArgs:\n    bucket_name: Name of the S3 bucket to optimize\n\nReturns:\n    Success status of the lifecycle policy application",
        "operationId": "apply_s3_lifecycle_optimization_api_cost_optimization_apply_s3_lifecycle__bucket_name__post",
        "parameters": [
          {
            "name": "bucket_name",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Bucket Name"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/cost-optimization/setup-monitoring": {
      "post": {
        "tags": [
          "cost-optimization"
        ],
        "summary": "Setup Cost Monitoring",
        "description": "Set up cost monitoring and budget alerts.\n\nReturns:\n    Success status of the cost monitoring setup",
        "operationId": "setup_cost_monitoring_api_cost_optimization_setup_monitoring_post",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/cost-optimization/config": {
      "get": {
        "tags": [
          "cost-optimization"
        ],
        "summary": "Get Cost Optimization Config",
        "description": "Get current cost optimization configuration.\n\nReturns:\n    Current cost optimization settings",
        "operationId": "get_cost_optimization_config_api_cost_optimization_config_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/health": {
      "get": {
        "tags": [
          "health"
        ],
        "summary": "Basic health check",
        "description": "Basic health check endpoint.",
        "operationId": "basic_health_check_api_health_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/health/detailed": {
      "get": {
        "tags": [
          "health"
        ],
        "summary": "Detailed health check",
        "description": "Detailed health check with all system components.",
        "operationId": "detailed_health_check_api_health_detailed_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/health/component/{component_name}": {
      "get": {
        "tags": [
          "health"
        ],
        "summary": "Component-specific health check",
        "description": "Get health status for a specific component.",
        "operationId": "component_health_check_api_health_component__component_name__get",
        "parameters": [
          {
            "name": "component_name",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Component Name"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/metrics": {
      "get": {
        "tags": [
          "health"
        ],
        "summary": "System metrics",
        "description": "Get current system metrics.",
        "operationId": "get_system_metrics_api_metrics_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/metrics/history": {
      "get": {
        "tags": [
          "health"
        ],
        "summary": "Historical metrics",
        "description": "Get historical system metrics.",
        "operationId": "get_metrics_history_api_metrics_history_get",
        "parameters": [
          {
            "name": "hours",
            "in": "query",
            "required": false,
            "schema": {
              "type": "integer",
              "maximum": 24,
              "minimum": 1,
              "description": "Hours of history to retrieve",
              "default": 1,
              "title": "Hours"
            },
            "description": "Hours of history to retrieve"
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/alerts": {
      "get": {
        "tags": [
          "health"
        ],
        "summary": "Current system alerts",
        "description": "Get current system alerts.",
        "operationId": "get_current_alerts_api_alerts_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/alerts/thresholds": {
      "post": {
        "tags": [
          "health"
        ],
        "summary": "Update alert thresholds",
        "description": "Update system alert thresholds.",
        "operationId": "update_alert_thresholds_api_alerts_thresholds_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "additionalProperties": {
                  "type": "number"
                },
                "type": "object",
                "title": "Thresholds"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/status": {
      "get": {
        "tags": [
          "health"
        ],
        "summary": "Overall system status",
        "description": "Get overall system status summary.",
        "operationId": "get_system_status_api_status_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/backup/full": {
      "post": {
        "tags": [
          "backup"
        ],
        "summary": "Create full system backup",
        "description": "Create a full system backup including database and files.",
        "operationId": "create_full_backup_api_backup_full_post",
        "parameters": [
          {
            "name": "include_files",
            "in": "query",
            "required": false,
            "schema": {
              "type": "boolean",
              "description": "Include uploaded files and generated content",
              "default": true,
              "title": "Include Files"
            },
            "description": "Include uploaded files and generated content"
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/backup/database": {
      "post": {
        "tags": [
          "backup"
        ],
        "summary": "Create database backup",
        "description": "Create a database-only backup.",
        "operationId": "create_database_backup_api_backup_database_post",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/backup/list": {
      "get": {
        "tags": [
          "backup"
        ],
        "summary": "List available backups",
        "description": "Get list of available backup files.",
        "operationId": "list_backups_api_backup_list_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/backup/download/{backup_name}": {
      "get": {
        "tags": [
          "backup"
        ],
        "summary": "Download backup file",
        "description": "Download a specific backup file.",
        "operationId": "download_backup_api_backup_download__backup_name__get",
        "parameters": [
          {
            "name": "backup_name",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Backup Name"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/backup/restore": {
      "post": {
        "tags": [
          "backup"
        ],
        "summary": "Restore from backup",
        "description": "Restore system from a backup file.",
        "operationId": "restore_from_backup_api_backup_restore_post",
        "parameters": [
          {
            "name": "backup_name",
            "in": "query",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Backup Name"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/backup/upload": {
      "post": {
        "tags": [
          "backup"
        ],
        "summary": "Upload backup file",
        "description": "Upload a backup file for restoration.",
        "operationId": "upload_backup_api_backup_upload_post",
        "requestBody": {
          "content": {
            "multipart/form-data": {
              "schema": {
                "$ref": "#/components/schemas/Body_upload_backup_api_backup_upload_post"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/export": {
      "post": {
        "tags": [
          "backup"
        ],
        "summary": "Export system data",
        "description": "Export system data in specified format.",
        "operationId": "export_data_api_export_post",
        "parameters": [
          {
            "name": "format",
            "in": "query",
            "required": false,
            "schema": {
              "type": "string",
              "pattern": "^(json|csv)$",
              "description": "Export format",
              "default": "json",
              "title": "Format"
            },
            "description": "Export format"
          },
          {
            "name": "include_embeddings",
            "in": "query",
            "required": false,
            "schema": {
              "type": "boolean",
              "description": "Include embedding vectors",
              "default": false,
              "title": "Include Embeddings"
            },
            "description": "Include embedding vectors"
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/export/download/{export_name}": {
      "get": {
        "tags": [
          "backup"
        ],
        "summary": "Download exported data",
        "description": "Download an exported data file.",
        "operationId": "download_export_api_export_download__export_name__get",
        "parameters": [
          {
            "name": "export_name",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Export Name"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/backup/{backup_name}": {
      "delete": {
        "tags": [
          "backup"
        ],
        "summary": "Delete backup file",
        "description": "Delete a specific backup file.",
        "operationId": "delete_backup_api_backup__backup_name__delete",
        "parameters": [
          {
            "name": "backup_name",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Backup Name"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/api/cleanup": {
      "post": {
        "tags": [
          "backup"
        ],
        "summary": "Clean up old backups",
        "description": "Clean up old backup files based on retention policy.",
        "operationId": "cleanup_backups_api_cleanup_post",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/storage/info": {
      "get": {
        "tags": [
          "backup"
        ],
        "summary": "Get storage information",
        "description": "Get information about storage usage and organization.",
        "operationId": "get_storage_info_api_storage_info_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api/storage/cleanup": {
      "post": {
        "tags": [
          "backup"
        ],
        "summary": "Clean up temporary and orphaned files",
        "description": "Clean up temporary files and orphaned data.",
        "operationId": "cleanup_storage_api_storage_cleanup_post",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/auth/login": {
      "get": {
        "tags": [
          "authentication"
        ],
        "summary": "Login Page",
        "description": "Display login page.\n\nArgs:\n    request: FastAPI request object\n    redirect: URL to redirect to after successful login\n\nReturns:\n    HTML login page",
        "operationId": "login_page_auth_login_get",
        "parameters": [
          {
            "name": "redirect",
            "in": "query",
            "required": false,
            "schema": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ],
              "title": "Redirect"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "text/html": {
                "schema": {
                  "type": "string"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      },
      "post": {
        "tags": [
          "authentication"
        ],
        "summary": "Login",
        "description": "Process login form submission.\n\nArgs:\n    request: FastAPI request object\n    response: FastAPI response object\n    password: Admin password\n    redirect_url: URL to redirect to after successful login\n\nReturns:\n    Redirect response or error page",
        "operationId": "login_auth_login_post",
        "requestBody": {
          "required": true,
          "content": {
            "application/x-www-form-urlencoded": {
              "schema": {
                "$ref": "#/components/schemas/Body_login_auth_login_post"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/auth/api/login": {
      "post": {
        "tags": [
          "authentication"
        ],
        "summary": "Api Login",
        "description": "API endpoint for programmatic login.\n\nArgs:\n    request: FastAPI request object\n    login_data: Login request data\n\nReturns:\n    Login response with session token",
        "operationId": "api_login_auth_api_login_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/LoginRequest"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/LoginResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/auth/logout": {
      "post": {
        "tags": [
          "authentication"
        ],
        "summary": "Logout",
        "description": "Logout user and invalidate session.\n\nArgs:\n    request: FastAPI request object\n    response: FastAPI response object\n\nReturns:\n    Redirect to home page",
        "operationId": "logout_auth_logout_post",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/auth/status": {
      "get": {
        "tags": [
          "authentication"
        ],
        "summary": "Session Status",
        "description": "Get current session status.\n\nArgs:\n    request: FastAPI request object\n\nReturns:\n    Session status information",
        "operationId": "session_status_auth_status_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SessionStatus"
                }
              }
            }
          }
        }
      }
    },
    "/auth/stats": {
      "get": {
        "tags": [
          "authentication"
        ],
        "summary": "Auth Stats",
        "description": "Get authentication service statistics (admin only).\n\nArgs:\n    request: FastAPI request object\n\nReturns:\n    Authentication statistics",
        "operationId": "auth_stats_auth_stats_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/demo/": {
      "get": {
        "tags": [
          "demo"
        ],
        "summary": "Get Demo Page",
        "description": "Get the complete demo page with all content.\n\nReturns:\n    HTML response with complete demo page content",
        "operationId": "get_demo_page_demo__get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "text/html": {
                "schema": {
                  "type": "string"
                }
              }
            }
          }
        }
      }
    },
    "/demo/samples": {
      "get": {
        "tags": [
          "demo"
        ],
        "summary": "Get Demo Samples",
        "description": "Get all demo samples with analysis results.\n\nReturns:\n    List of demo samples with complete analysis data",
        "operationId": "get_demo_samples_demo_samples_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SuccessResponse"
                }
              }
            }
          }
        }
      }
    },
    "/demo/samples/{sample_id}": {
      "get": {
        "tags": [
          "demo"
        ],
        "summary": "Get Demo Sample",
        "description": "Get a specific demo sample by ID.\n\nArgs:\n    sample_id: ID of the demo sample\n\nReturns:\n    Demo sample with complete analysis data",
        "operationId": "get_demo_sample_demo_samples__sample_id__get",
        "parameters": [
          {
            "name": "sample_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer",
              "title": "Sample Id"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SuccessResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/demo/project-info": {
      "get": {
        "tags": [
          "demo"
        ],
        "summary": "Get Project Info",
        "description": "Get comprehensive project information for demo page.\n\nReturns:\n    Project description with technical details and features",
        "operationId": "get_project_info_demo_project_info_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SuccessResponse"
                }
              }
            }
          }
        }
      }
    },
    "/demo/disclaimer": {
      "get": {
        "tags": [
          "demo"
        ],
        "summary": "Get Medical Disclaimer",
        "description": "Get medical disclaimer and warnings for demo content.\n\nReturns:\n    Medical disclaimer with all required warnings",
        "operationId": "get_medical_disclaimer_demo_disclaimer_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SuccessResponse"
                }
              }
            }
          }
        }
      }
    },
    "/demo/technical-links": {
      "get": {
        "tags": [
          "demo"
        ],
        "summary": "Get Technical Links",
        "description": "Get technical links and documentation references.\n\nReturns:\n    Technical links including GitHub repository and documentation",
        "operationId": "get_technical_links_demo_technical_links_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SuccessResponse"
                }
              }
            }
          }
        }
      }
    },
    "/demo/statistics": {
      "get": {
        "tags": [
          "demo"
        ],
        "summary": "Get Demo Statistics",
        "description": "Get demo-specific statistics and metrics.\n\nReturns:\n    Demo statistics including sample counts and distributions",
        "operationId": "get_demo_statistics_demo_statistics_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SuccessResponse"
                }
              }
            }
          }
        }
      }
    },
    "/": {
      "get": {
        "summary": "Root Fallback",
        "description": "Fallback root endpoint when React frontend build is not available.",
        "operationId": "root_fallback__get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/api": {
      "get": {
        "summary": "Api Root",
        "description": "API root endpoint - returns basic API information.",
        "operationId": "api_root_api_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/health/detailed": {
      "get": {
        "summary": "Detailed Health Check",
        "description": "Detailed health check with system information.",
        "operationId": "detailed_health_check_health_detailed_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/metrics": {
      "get": {
        "summary": "Get Metrics",
        "description": "Get system metrics for monitoring.",
        "operationId": "get_metrics_metrics_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          }
        }
      }
    },
    "/monitoring/logs": {
      "get": {
        "summary": "Get Recent Logs",
        "description": "Get recent structured logs for monitoring.",
        "operationId": "get_recent_logs_monitoring_logs_get",
        "parameters": [
          {
            "name": "limit",
            "in": "query",
            "required": false,
            "schema": {
              "type": "integer",
              "default": 100,
              "title": "Limit"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/monitoring/alerts": {
      "get": {
        "summary": "Get Recent Alerts",
        "description": "Get recent alerts for monitoring.",
        "operationId": "get_recent_alerts_monitoring_alerts_get",
        "parameters": [
          {
            "name": "limit",
            "in": "query",
            "required": false,
            "schema": {
              "type": "integer",
              "default": 50,
              "title": "Limit"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    }
  },
  "components": {
    "schemas": {
      "AgeGroupModelResponse": {
        "properties": {
          "id": {
            "type": "integer",
            "title": "Id"
          },
          "age_min": {
            "type": "number",
            "maximum": 18.0,
            "minimum": 2.0,
            "title": "Age Min"
          },
          "age_max": {
            "type": "number",
            "maximum": 18.0,
            "minimum": 2.0,
            "title": "Age Max"
          },
          "model_type": {
            "$ref": "#/components/schemas/AnalysisMethod"
          },
          "vision_model": {
            "$ref": "#/components/schemas/VisionModel"
          },
          "sample_count": {
            "type": "integer",
            "minimum": 0.0,
            "title": "Sample Count"
          },
          "threshold": {
            "type": "number",
            "exclusiveMinimum": 0.0,
            "title": "Threshold"
          },
          "status": {
            "$ref": "#/components/schemas/ModelStatus"
          },
          "created_timestamp": {
            "type": "string",
            "format": "date-time",
            "title": "Created Timestamp"
          },
          "is_active": {
            "type": "boolean",
            "title": "Is Active"
          }
        },
        "type": "object",
        "required": [
          "id",
          "age_min",
          "age_max",
          "model_type",
          "vision_model",
          "sample_count",
          "threshold",
          "status",
          "created_timestamp",
          "is_active"
        ],
        "title": "AgeGroupModelResponse",
        "description": "Response model for age group model information."
      },
      "AgeGroupingStrategy": {
        "type": "string",
        "enum": [
          "yearly",
          "custom"
        ],
        "title": "AgeGroupingStrategy",
        "description": "Enumeration for age grouping strategies."
      },
      "AnalysisHistoryResponse": {
        "properties": {
          "drawing_id": {
            "type": "integer",
            "title": "Drawing Id"
          },
          "analyses": {
            "items": {
              "$ref": "#/components/schemas/AnomalyAnalysisResponse"
            },
            "type": "array",
            "title": "Analyses"
          },
          "total_count": {
            "type": "integer",
            "title": "Total Count"
          }
        },
        "type": "object",
        "required": [
          "drawing_id",
          "analyses",
          "total_count"
        ],
        "title": "AnalysisHistoryResponse",
        "description": "Response model for analysis history of a drawing."
      },
      "AnalysisMethod": {
        "type": "string",
        "enum": [
          "autoencoder"
        ],
        "title": "AnalysisMethod",
        "description": "Enumeration for anomaly detection methods."
      },
      "AnalysisRequest": {
        "properties": {
          "force_reanalysis": {
            "type": "boolean",
            "title": "Force Reanalysis",
            "description": "Force re-analysis even if results exist",
            "default": false
          }
        },
        "type": "object",
        "title": "AnalysisRequest",
        "description": "Request model for analyzing a drawing."
      },
      "AnalysisResultResponse": {
        "properties": {
          "drawing": {
            "$ref": "#/components/schemas/DrawingResponse"
          },
          "analysis": {
            "$ref": "#/components/schemas/AnomalyAnalysisResponse"
          },
          "interpretability": {
            "anyOf": [
              {
                "$ref": "#/components/schemas/InterpretabilityResponse"
              },
              {
                "type": "null"
              }
            ]
          },
          "comparison_examples": {
            "items": {
              "$ref": "#/components/schemas/ComparisonExampleResponse"
            },
            "type": "array",
            "title": "Comparison Examples",
            "description": "Similar normal examples from the same age group"
          }
        },
        "type": "object",
        "required": [
          "drawing",
          "analysis"
        ],
        "title": "AnalysisResultResponse",
        "description": "Complete analysis result including drawing, analysis, and interpretability."
      },
      "AnnotationRequest": {
        "properties": {
          "region_id": {
            "type": "string",
            "title": "Region Id",
            "description": "ID of the region being annotated"
          },
          "annotation_text": {
            "type": "string",
            "maxLength": 500,
            "minLength": 1,
            "title": "Annotation Text",
            "description": "User annotation text"
          },
          "annotation_type": {
            "type": "string",
            "title": "Annotation Type",
            "description": "Type: note, question, concern, etc."
          },
          "user_id": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "User Id",
            "description": "Optional user identifier"
          }
        },
        "type": "object",
        "required": [
          "region_id",
          "annotation_text",
          "annotation_type"
        ],
        "title": "AnnotationRequest",
        "description": "Request model for adding annotations to interpretability results."
      },
      "AnomalyAnalysisResponse": {
        "properties": {
          "id": {
            "type": "integer",
            "title": "Id"
          },
          "drawing_id": {
            "type": "integer",
            "title": "Drawing Id"
          },
          "anomaly_score": {
            "type": "number",
            "title": "Anomaly Score",
            "description": "Overall reconstruction loss on full 832-dimensional embedding"
          },
          "normalized_score": {
            "type": "number",
            "maximum": 100.0,
            "minimum": 0.0,
            "title": "Normalized Score",
            "description": "Normalized anomaly score (0-100 scale: 0=no anomaly, 100=maximal anomaly)"
          },
          "visual_anomaly_score": {
            "anyOf": [
              {
                "type": "number"
              },
              {
                "type": "null"
              }
            ],
            "title": "Visual Anomaly Score",
            "description": "Visual component reconstruction loss (dims 0-767)"
          },
          "subject_anomaly_score": {
            "anyOf": [
              {
                "type": "number"
              },
              {
                "type": "null"
              }
            ],
            "title": "Subject Anomaly Score",
            "description": "Subject component reconstruction loss (dims 768-831)"
          },
          "anomaly_attribution": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "Anomaly Attribution",
            "description": "Primary anomaly source: 'visual', 'subject', 'both', or 'age'"
          },
          "analysis_type": {
            "type": "string",
            "title": "Analysis Type",
            "description": "Type of analysis performed",
            "default": "subject_aware"
          },
          "subject_category": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "Subject Category",
            "description": "Subject category used in analysis"
          },
          "is_anomaly": {
            "type": "boolean",
            "title": "Is Anomaly",
            "description": "Whether the drawing is flagged as anomalous"
          },
          "confidence": {
            "type": "number",
            "maximum": 1.0,
            "minimum": 0.0,
            "title": "Confidence",
            "description": "Confidence in the anomaly decision"
          },
          "age_group": {
            "type": "string",
            "title": "Age Group",
            "description": "Age group used for analysis"
          },
          "method_used": {
            "$ref": "#/components/schemas/AnalysisMethod",
            "description": "Analysis method used"
          },
          "vision_model": {
            "$ref": "#/components/schemas/VisionModel",
            "description": "Vision model used"
          },
          "analysis_timestamp": {
            "type": "string",
            "format": "date-time",
            "title": "Analysis Timestamp"
          }
        },
        "type": "object",
        "required": [
          "id",
          "drawing_id",
          "anomaly_score",
          "normalized_score",
          "is_anomaly",
          "confidence",
          "age_group",
          "method_used",
          "vision_model",
          "analysis_timestamp"
        ],
        "title": "AnomalyAnalysisResponse",
        "description": "Response model for anomaly analysis results."
      },
      "AttentionPatchResponse": {
        "properties": {
          "patch_id": {
            "type": "string",
            "title": "Patch Id",
            "description": "Unique identifier for the patch"
          },
          "coordinates": {
            "items": {
              "type": "integer"
            },
            "type": "array",
            "title": "Coordinates",
            "description": "Patch coordinates [x, y, width, height]"
          },
          "attention_weight": {
            "type": "number",
            "maximum": 1.0,
            "minimum": 0.0,
            "title": "Attention Weight",
            "description": "Attention weight for this patch"
          },
          "layer_index": {
            "type": "integer",
            "minimum": 0.0,
            "title": "Layer Index",
            "description": "Transformer layer index"
          },
          "head_index": {
            "type": "integer",
            "minimum": 0.0,
            "title": "Head Index",
            "description": "Attention head index"
          }
        },
        "type": "object",
        "required": [
          "patch_id",
          "coordinates",
          "attention_weight",
          "layer_index",
          "head_index"
        ],
        "title": "AttentionPatchResponse",
        "description": "Response model for Vision Transformer attention patches."
      },
      "BackupRequest": {
        "properties": {
          "upload_to_s3": {
            "anyOf": [
              {
                "type": "boolean"
              },
              {
                "type": "null"
              }
            ],
            "title": "Upload To S3"
          },
          "include_files": {
            "type": "boolean",
            "title": "Include Files",
            "default": false
          }
        },
        "type": "object",
        "title": "BackupRequest",
        "description": "Request model for database backup operations"
      },
      "BatchAnalysisRequest": {
        "properties": {
          "drawing_ids": {
            "items": {
              "type": "integer"
            },
            "type": "array",
            "maxItems": 100,
            "minItems": 1,
            "title": "Drawing Ids",
            "description": "List of drawing IDs to analyze"
          },
          "force_reanalysis": {
            "type": "boolean",
            "title": "Force Reanalysis",
            "description": "Force re-analysis even if results exist",
            "default": false
          }
        },
        "type": "object",
        "required": [
          "drawing_ids"
        ],
        "title": "BatchAnalysisRequest",
        "description": "Request model for batch analysis of multiple drawings."
      },
      "BatchAnalysisResponse": {
        "properties": {
          "batch_id": {
            "type": "string",
            "title": "Batch Id",
            "description": "Unique identifier for the batch"
          },
          "total_drawings": {
            "type": "integer",
            "exclusiveMinimum": 0.0,
            "title": "Total Drawings",
            "description": "Total number of drawings to analyze"
          },
          "completed": {
            "type": "integer",
            "minimum": 0.0,
            "title": "Completed",
            "description": "Number of completed analyses"
          },
          "failed": {
            "type": "integer",
            "minimum": 0.0,
            "title": "Failed",
            "description": "Number of failed analyses"
          },
          "status": {
            "type": "string",
            "title": "Status",
            "description": "Batch processing status"
          },
          "results": {
            "items": {
              "$ref": "#/components/schemas/AnalysisResultResponse"
            },
            "type": "array",
            "title": "Results",
            "description": "Completed analysis results"
          },
          "errors": {
            "items": {
              "additionalProperties": true,
              "type": "object"
            },
            "type": "array",
            "title": "Errors",
            "description": "Error details for failed analyses"
          },
          "started_at": {
            "type": "string",
            "format": "date-time",
            "title": "Started At"
          },
          "completed_at": {
            "anyOf": [
              {
                "type": "string",
                "format": "date-time"
              },
              {
                "type": "null"
              }
            ],
            "title": "Completed At"
          }
        },
        "type": "object",
        "required": [
          "batch_id",
          "total_drawings",
          "completed",
          "failed",
          "status",
          "started_at"
        ],
        "title": "BatchAnalysisResponse",
        "description": "Response model for batch analysis operations."
      },
      "Body_batch_upload_drawings_api_drawings_batch_upload_post": {
        "properties": {
          "files": {
            "items": {
              "type": "string",
              "format": "binary"
            },
            "type": "array",
            "title": "Files",
            "description": "Multiple drawing files"
          }
        },
        "type": "object",
        "required": [
          "files"
        ],
        "title": "Body_batch_upload_drawings_api_drawings_batch_upload_post"
      },
      "Body_batch_upload_drawings_api_v1_drawings_batch_upload_post": {
        "properties": {
          "files": {
            "items": {
              "type": "string",
              "format": "binary"
            },
            "type": "array",
            "title": "Files",
            "description": "Multiple drawing files"
          }
        },
        "type": "object",
        "required": [
          "files"
        ],
        "title": "Body_batch_upload_drawings_api_v1_drawings_batch_upload_post"
      },
      "Body_login_auth_login_post": {
        "properties": {
          "password": {
            "type": "string",
            "title": "Password"
          },
          "redirect_url": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "Redirect Url"
          }
        },
        "type": "object",
        "required": [
          "password"
        ],
        "title": "Body_login_auth_login_post"
      },
      "Body_merge_age_groups_api_models_data_sufficiency_merge_age_groups_post": {
        "properties": {
          "original_groups": {
            "items": {
              "items": {
                "type": "number"
              },
              "type": "array"
            },
            "type": "array",
            "title": "Original Groups"
          },
          "merged_group": {
            "items": {
              "type": "number"
            },
            "type": "array",
            "title": "Merged Group"
          }
        },
        "type": "object",
        "required": [
          "original_groups",
          "merged_group"
        ],
        "title": "Body_merge_age_groups_api_models_data_sufficiency_merge_age_groups_post"
      },
      "Body_merge_age_groups_api_v1_models_data_sufficiency_merge_age_groups_post": {
        "properties": {
          "original_groups": {
            "items": {
              "items": {
                "type": "number"
              },
              "type": "array"
            },
            "type": "array",
            "title": "Original Groups"
          },
          "merged_group": {
            "items": {
              "type": "number"
            },
            "type": "array",
            "title": "Merged Group"
          }
        },
        "type": "object",
        "required": [
          "original_groups",
          "merged_group"
        ],
        "title": "Body_merge_age_groups_api_v1_models_data_sufficiency_merge_age_groups_post"
      },
      "Body_upload_backup_api_backup_upload_post": {
        "properties": {
          "file": {
            "type": "string",
            "format": "binary",
            "title": "File"
          }
        },
        "type": "object",
        "required": [
          "file"
        ],
        "title": "Body_upload_backup_api_backup_upload_post"
      },
      "Body_upload_backup_api_v1_backup_upload_post": {
        "properties": {
          "file": {
            "type": "string",
            "format": "binary",
            "title": "File"
          }
        },
        "type": "object",
        "required": [
          "file"
        ],
        "title": "Body_upload_backup_api_v1_backup_upload_post"
      },
      "Body_upload_drawing_api_drawings_upload_post": {
        "properties": {
          "file": {
            "type": "string",
            "format": "binary",
            "title": "File",
            "description": "Drawing image file (PNG, JPEG, BMP)"
          },
          "age_years": {
            "type": "number",
            "maximum": 18.0,
            "minimum": 2.0,
            "title": "Age Years",
            "description": "Child's age in years"
          },
          "subject": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "Subject",
            "description": "Drawing subject"
          },
          "expert_label": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "Expert Label",
            "description": "Expert assessment"
          },
          "drawing_tool": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "Drawing Tool",
            "description": "Drawing tool used"
          },
          "prompt": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "Prompt",
            "description": "Drawing prompt"
          }
        },
        "type": "object",
        "required": [
          "file",
          "age_years"
        ],
        "title": "Body_upload_drawing_api_drawings_upload_post"
      },
      "Body_upload_drawing_api_v1_drawings_upload_post": {
        "properties": {
          "file": {
            "type": "string",
            "format": "binary",
            "title": "File",
            "description": "Drawing image file (PNG, JPEG, BMP)"
          },
          "age_years": {
            "type": "number",
            "maximum": 18.0,
            "minimum": 2.0,
            "title": "Age Years",
            "description": "Child's age in years"
          },
          "subject": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "Subject",
            "description": "Drawing subject"
          },
          "expert_label": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "Expert Label",
            "description": "Expert assessment"
          },
          "drawing_tool": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "Drawing Tool",
            "description": "Drawing tool used"
          },
          "prompt": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "Prompt",
            "description": "Drawing prompt"
          }
        },
        "type": "object",
        "required": [
          "file",
          "age_years"
        ],
        "title": "Body_upload_drawing_api_v1_drawings_upload_post"
      },
      "ComparisonExampleResponse": {
        "properties": {
          "drawing_id": {
            "type": "integer",
            "title": "Drawing Id"
          },
          "filename": {
            "type": "string",
            "title": "Filename"
          },
          "age_years": {
            "type": "number",
            "title": "Age Years"
          },
          "subject": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "Subject"
          },
          "similarity_score": {
            "type": "number",
            "maximum": 1.0,
            "minimum": 0.0,
            "title": "Similarity Score",
            "description": "Similarity score to the analyzed drawing"
          },
          "anomaly_score": {
            "type": "number",
            "title": "Anomaly Score",
            "description": "Anomaly score of the comparison example"
          },
          "normalized_score": {
            "type": "number",
            "maximum": 100.0,
            "minimum": 0.0,
            "title": "Normalized Score",
            "description": "Normalized anomaly score of the comparison example (0-100 scale: 0=no anomaly, 100=maximal anomaly)"
          }
        },
        "type": "object",
        "required": [
          "drawing_id",
          "filename",
          "age_years",
          "similarity_score",
          "anomaly_score",
          "normalized_score"
        ],
        "title": "ComparisonExampleResponse",
        "description": "Response model for comparison examples."
      },
      "ComparisonExamplesResponse": {
        "properties": {
          "normal_examples": {
            "items": {
              "additionalProperties": true,
              "type": "object"
            },
            "type": "array",
            "title": "Normal Examples",
            "description": "Normal drawings from same age group"
          },
          "anomalous_examples": {
            "items": {
              "additionalProperties": true,
              "type": "object"
            },
            "type": "array",
            "title": "Anomalous Examples",
            "description": "Other anomalous examples"
          },
          "explanation_context": {
            "type": "string",
            "title": "Explanation Context",
            "description": "Context for the comparisons"
          },
          "age_group": {
            "type": "string",
            "title": "Age Group",
            "description": "Age group for the examples"
          },
          "total_available": {
            "type": "integer",
            "title": "Total Available",
            "description": "Total examples available in this age group"
          }
        },
        "type": "object",
        "required": [
          "normal_examples",
          "anomalous_examples",
          "explanation_context",
          "age_group",
          "total_available"
        ],
        "title": "ComparisonExamplesResponse",
        "description": "Response model for comparison examples from the same age group."
      },
      "ConfidenceMetricsResponse": {
        "properties": {
          "overall_confidence": {
            "type": "number",
            "maximum": 1.0,
            "minimum": 0.0,
            "title": "Overall Confidence",
            "description": "Overall confidence in the analysis"
          },
          "explanation_reliability": {
            "type": "number",
            "maximum": 1.0,
            "minimum": 0.0,
            "title": "Explanation Reliability",
            "description": "Reliability of the explanation"
          },
          "model_certainty": {
            "type": "number",
            "maximum": 1.0,
            "minimum": 0.0,
            "title": "Model Certainty",
            "description": "Model's certainty in the prediction"
          },
          "data_sufficiency": {
            "type": "string",
            "title": "Data Sufficiency",
            "description": "Sufficient/Limited/Insufficient data quality"
          },
          "warnings": {
            "items": {
              "type": "string"
            },
            "type": "array",
            "title": "Warnings",
            "description": "Confidence-related warnings"
          },
          "technical_details": {
            "additionalProperties": true,
            "type": "object",
            "title": "Technical Details",
            "description": "Technical confidence metrics"
          }
        },
        "type": "object",
        "required": [
          "overall_confidence",
          "explanation_reliability",
          "model_certainty",
          "data_sufficiency"
        ],
        "title": "ConfidenceMetricsResponse",
        "description": "Response model for confidence metrics and reliability scores."
      },
      "ConfigurationUpdateRequest": {
        "properties": {
          "threshold_percentile": {
            "anyOf": [
              {
                "type": "number",
                "maximum": 99.9,
                "minimum": 50.0
              },
              {
                "type": "null"
              }
            ],
            "title": "Threshold Percentile",
            "description": "Percentile for threshold calculation"
          },
          "age_grouping_strategy": {
            "anyOf": [
              {
                "$ref": "#/components/schemas/AgeGroupingStrategy"
              },
              {
                "type": "null"
              }
            ]
          },
          "min_samples_per_group": {
            "anyOf": [
              {
                "type": "integer",
                "minimum": 10.0
              },
              {
                "type": "null"
              }
            ],
            "title": "Min Samples Per Group"
          },
          "max_age_group_span": {
            "anyOf": [
              {
                "type": "number",
                "maximum": 16.0,
                "exclusiveMinimum": 0.0
              },
              {
                "type": "null"
              }
            ],
            "title": "Max Age Group Span"
          }
        },
        "type": "object",
        "title": "ConfigurationUpdateRequest",
        "description": "Request model for updating system configuration."
      },
      "ConsistencyCheckRequest": {
        "properties": {
          "other_db_url": {
            "type": "string",
            "title": "Other Db Url"
          }
        },
        "type": "object",
        "required": [
          "other_db_url"
        ],
        "title": "ConsistencyCheckRequest",
        "description": "Request model for cross-environment consistency checks"
      },
      "CostComplianceResponse": {
        "properties": {
          "is_compliant": {
            "type": "boolean",
            "title": "Is Compliant"
          },
          "total_estimated_cost": {
            "type": "number",
            "title": "Total Estimated Cost"
          },
          "budget_limit": {
            "type": "number",
            "title": "Budget Limit"
          },
          "target_range": {
            "additionalProperties": {
              "type": "number"
            },
            "type": "object",
            "title": "Target Range"
          },
          "cost_breakdown": {
            "items": {
              "additionalProperties": true,
              "type": "object"
            },
            "type": "array",
            "title": "Cost Breakdown"
          },
          "recommendations": {
            "items": {
              "type": "string"
            },
            "type": "array",
            "title": "Recommendations"
          }
        },
        "type": "object",
        "required": [
          "is_compliant",
          "total_estimated_cost",
          "budget_limit",
          "target_range",
          "cost_breakdown",
          "recommendations"
        ],
        "title": "CostComplianceResponse",
        "description": "Response model for cost compliance validation."
      },
      "CostEstimateResponse": {
        "properties": {
          "total_monthly_cost": {
            "type": "number",
            "title": "Total Monthly Cost"
          },
          "is_within_budget": {
            "type": "boolean",
            "title": "Is Within Budget"
          },
          "cost_breakdown": {
            "items": {
              "$ref": "#/components/schemas/ResourceCostEstimate"
            },
            "type": "array",
            "title": "Cost Breakdown"
          },
          "target_range": {
            "additionalProperties": {
              "type": "number"
            },
            "type": "object",
            "title": "Target Range"
          }
        },
        "type": "object",
        "required": [
          "total_monthly_cost",
          "is_within_budget",
          "cost_breakdown",
          "target_range"
        ],
        "title": "CostEstimateResponse",
        "description": "Response model for cost estimates."
      },
      "CostOptimizationResponse": {
        "properties": {
          "ecs_fargate_config": {
            "additionalProperties": {
              "type": "integer"
            },
            "type": "object",
            "title": "Ecs Fargate Config"
          },
          "s3_lifecycle_policy": {
            "additionalProperties": true,
            "type": "object",
            "title": "S3 Lifecycle Policy"
          },
          "cloudfront_cache_config": {
            "additionalProperties": true,
            "type": "object",
            "title": "Cloudfront Cache Config"
          },
          "recommendations": {
            "items": {
              "type": "string"
            },
            "type": "array",
            "title": "Recommendations"
          }
        },
        "type": "object",
        "required": [
          "ecs_fargate_config",
          "s3_lifecycle_policy",
          "cloudfront_cache_config",
          "recommendations"
        ],
        "title": "CostOptimizationResponse",
        "description": "Response model for cost optimization recommendations."
      },
      "DocumentationMetrics": {
        "properties": {
          "total_files": {
            "type": "integer",
            "title": "Total Files",
            "description": "Total number of documentation files"
          },
          "last_generated": {
            "anyOf": [
              {
                "type": "string",
                "format": "date-time"
              },
              {
                "type": "null"
              }
            ],
            "title": "Last Generated",
            "description": "Last generation timestamp"
          },
          "generation_count": {
            "type": "integer",
            "title": "Generation Count",
            "description": "Total number of generations"
          },
          "average_duration": {
            "type": "number",
            "title": "Average Duration",
            "description": "Average generation duration in seconds"
          },
          "success_rate": {
            "type": "number",
            "title": "Success Rate",
            "description": "Success rate percentage"
          },
          "file_breakdown": {
            "additionalProperties": {
              "type": "integer"
            },
            "type": "object",
            "title": "File Breakdown",
            "description": "Files by category"
          },
          "validation_status": {
            "additionalProperties": true,
            "type": "object",
            "title": "Validation Status",
            "description": "Validation results"
          }
        },
        "type": "object",
        "required": [
          "total_files",
          "generation_count",
          "average_duration",
          "success_rate"
        ],
        "title": "DocumentationMetrics",
        "description": "Documentation metrics model."
      },
      "DocumentationStatus": {
        "properties": {
          "is_generating": {
            "type": "boolean",
            "title": "Is Generating",
            "description": "Whether documentation is currently being generated"
          },
          "current_task": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "Current Task",
            "description": "Current generation task"
          },
          "progress": {
            "type": "integer",
            "title": "Progress",
            "description": "Generation progress percentage (0-100)",
            "default": 0
          },
          "start_time": {
            "anyOf": [
              {
                "type": "string",
                "format": "date-time"
              },
              {
                "type": "null"
              }
            ],
            "title": "Start Time",
            "description": "Generation start time"
          },
          "last_update": {
            "anyOf": [
              {
                "type": "string",
                "format": "date-time"
              },
              {
                "type": "null"
              }
            ],
            "title": "Last Update",
            "description": "Last status update time"
          },
          "errors": {
            "items": {
              "type": "string"
            },
            "type": "array",
            "title": "Errors",
            "description": "Generation errors"
          },
          "warnings": {
            "items": {
              "type": "string"
            },
            "type": "array",
            "title": "Warnings",
            "description": "Generation warnings"
          }
        },
        "type": "object",
        "required": [
          "is_generating"
        ],
        "title": "DocumentationStatus",
        "description": "Documentation generation status model."
      },
      "DrawingListResponse": {
        "properties": {
          "drawings": {
            "items": {
              "$ref": "#/components/schemas/DrawingResponse"
            },
            "type": "array",
            "title": "Drawings"
          },
          "total_count": {
            "type": "integer",
            "title": "Total Count"
          },
          "page": {
            "type": "integer",
            "minimum": 1.0,
            "title": "Page",
            "description": "Current page number"
          },
          "page_size": {
            "type": "integer",
            "maximum": 100.0,
            "minimum": 1.0,
            "title": "Page Size",
            "description": "Number of items per page"
          },
          "total_pages": {
            "type": "integer",
            "title": "Total Pages"
          }
        },
        "type": "object",
        "required": [
          "drawings",
          "total_count",
          "page",
          "page_size",
          "total_pages"
        ],
        "title": "DrawingListResponse",
        "description": "Response model for listing multiple drawings."
      },
      "DrawingResponse": {
        "properties": {
          "id": {
            "type": "integer",
            "title": "Id"
          },
          "filename": {
            "type": "string",
            "title": "Filename"
          },
          "age_years": {
            "type": "number",
            "title": "Age Years"
          },
          "subject": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "Subject"
          },
          "expert_label": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "Expert Label"
          },
          "drawing_tool": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "Drawing Tool"
          },
          "prompt": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "Prompt"
          },
          "upload_timestamp": {
            "type": "string",
            "format": "date-time",
            "title": "Upload Timestamp"
          }
        },
        "type": "object",
        "required": [
          "id",
          "filename",
          "age_years",
          "subject",
          "expert_label",
          "drawing_tool",
          "prompt",
          "upload_timestamp"
        ],
        "title": "DrawingResponse",
        "description": "Response model for drawing information."
      },
      "ExpertLabel": {
        "type": "string",
        "enum": [
          "normal",
          "concern",
          "severe"
        ],
        "title": "ExpertLabel",
        "description": "Enumeration for expert labels on drawings."
      },
      "ExportRequest": {
        "properties": {
          "format": {
            "type": "string",
            "title": "Format",
            "description": "Export format: pdf, png, csv, json, html"
          },
          "include_annotations": {
            "type": "boolean",
            "title": "Include Annotations",
            "description": "Include user annotations",
            "default": true
          },
          "include_comparisons": {
            "type": "boolean",
            "title": "Include Comparisons",
            "description": "Include comparison examples",
            "default": true
          },
          "simplified_version": {
            "type": "boolean",
            "title": "Simplified Version",
            "description": "Use simplified explanations",
            "default": false
          },
          "export_options": {
            "additionalProperties": true,
            "type": "object",
            "title": "Export Options",
            "description": "Additional export options"
          }
        },
        "type": "object",
        "required": [
          "format"
        ],
        "title": "ExportRequest",
        "description": "Request model for exporting interpretability results."
      },
      "ExportResponse": {
        "properties": {
          "export_id": {
            "type": "string",
            "title": "Export Id",
            "description": "Unique identifier for the export"
          },
          "file_path": {
            "type": "string",
            "title": "File Path",
            "description": "Path to the exported file"
          },
          "file_url": {
            "type": "string",
            "title": "File Url",
            "description": "URL to download the exported file"
          },
          "format": {
            "type": "string",
            "title": "Format",
            "description": "Export format used"
          },
          "file_size": {
            "type": "integer",
            "title": "File Size",
            "description": "File size in bytes"
          },
          "created_at": {
            "type": "string",
            "format": "date-time",
            "title": "Created At",
            "description": "Export creation timestamp"
          },
          "expires_at": {
            "anyOf": [
              {
                "type": "string",
                "format": "date-time"
              },
              {
                "type": "null"
              }
            ],
            "title": "Expires At",
            "description": "Export expiration timestamp"
          }
        },
        "type": "object",
        "required": [
          "export_id",
          "file_path",
          "file_url",
          "format",
          "file_size",
          "created_at"
        ],
        "title": "ExportResponse",
        "description": "Response model for export operations."
      },
      "GenerationRequest": {
        "properties": {
          "categories": {
            "anyOf": [
              {
                "items": {
                  "type": "string"
                },
                "type": "array"
              },
              {
                "type": "null"
              }
            ],
            "title": "Categories",
            "description": "Specific categories to generate"
          },
          "force": {
            "type": "boolean",
            "title": "Force",
            "description": "Force regeneration even if no changes detected",
            "default": false
          },
          "validate_after": {
            "type": "boolean",
            "title": "Validate After",
            "description": "Run validation after generation",
            "default": true
          }
        },
        "type": "object",
        "title": "GenerationRequest",
        "description": "Documentation generation request model."
      },
      "GenerationResult": {
        "properties": {
          "success": {
            "type": "boolean",
            "title": "Success",
            "description": "Whether generation was successful"
          },
          "duration": {
            "type": "number",
            "title": "Duration",
            "description": "Generation duration in seconds"
          },
          "generated_files": {
            "items": {
              "type": "string"
            },
            "type": "array",
            "title": "Generated Files",
            "description": "List of generated files"
          },
          "errors": {
            "items": {
              "type": "string"
            },
            "type": "array",
            "title": "Errors",
            "description": "Generation errors"
          },
          "warnings": {
            "items": {
              "type": "string"
            },
            "type": "array",
            "title": "Warnings",
            "description": "Generation warnings"
          },
          "validation_result": {
            "anyOf": [
              {
                "additionalProperties": true,
                "type": "object"
              },
              {
                "type": "null"
              }
            ],
            "title": "Validation Result",
            "description": "Validation results"
          }
        },
        "type": "object",
        "required": [
          "success",
          "duration"
        ],
        "title": "GenerationResult",
        "description": "Documentation generation result model."
      },
      "HTTPValidationError": {
        "properties": {
          "detail": {
            "items": {
              "$ref": "#/components/schemas/ValidationError"
            },
            "type": "array",
            "title": "Detail"
          }
        },
        "type": "object",
        "title": "HTTPValidationError"
      },
      "HealthCheckResponse": {
        "properties": {
          "status": {
            "type": "string",
            "title": "Status",
            "description": "Service status"
          },
          "timestamp": {
            "type": "string",
            "title": "Timestamp",
            "description": "Check timestamp"
          },
          "version": {
            "type": "string",
            "title": "Version",
            "description": "Application version"
          },
          "database": {
            "type": "string",
            "title": "Database",
            "description": "Database status"
          },
          "models": {
            "type": "string",
            "title": "Models",
            "description": "ML models status"
          },
          "storage": {
            "type": "string",
            "title": "Storage",
            "description": "File storage status"
          }
        },
        "type": "object",
        "required": [
          "status",
          "timestamp",
          "version",
          "database",
          "models",
          "storage"
        ],
        "title": "HealthCheckResponse",
        "description": "Health check response model."
      },
      "InteractiveInterpretabilityResponse": {
        "properties": {
          "saliency_regions": {
            "items": {
              "$ref": "#/components/schemas/InteractiveRegionResponse"
            },
            "type": "array",
            "title": "Saliency Regions",
            "description": "Interactive regions with hover explanations"
          },
          "attention_patches": {
            "items": {
              "$ref": "#/components/schemas/AttentionPatchResponse"
            },
            "type": "array",
            "title": "Attention Patches",
            "description": "Vision Transformer attention patch data"
          },
          "region_explanations": {
            "additionalProperties": {
              "type": "string"
            },
            "type": "object",
            "title": "Region Explanations",
            "description": "Explanations for each interactive region"
          },
          "confidence_scores": {
            "additionalProperties": {
              "type": "number"
            },
            "type": "object",
            "title": "Confidence Scores",
            "description": "Confidence scores for each explanation"
          },
          "interaction_metadata": {
            "additionalProperties": true,
            "type": "object",
            "title": "Interaction Metadata",
            "description": "Metadata for interactive features"
          }
        },
        "type": "object",
        "required": [
          "saliency_regions",
          "attention_patches",
          "region_explanations",
          "confidence_scores"
        ],
        "title": "InteractiveInterpretabilityResponse",
        "description": "Response model for interactive interpretability data."
      },
      "InteractiveRegionResponse": {
        "properties": {
          "region_id": {
            "type": "string",
            "title": "Region Id",
            "description": "Unique identifier for the region"
          },
          "bounding_box": {
            "items": {
              "type": "integer"
            },
            "type": "array",
            "title": "Bounding Box",
            "description": "Bounding box coordinates [x1, y1, x2, y2]"
          },
          "importance_score": {
            "type": "number",
            "maximum": 1.0,
            "minimum": 0.0,
            "title": "Importance Score",
            "description": "Importance score for this region"
          },
          "spatial_location": {
            "type": "string",
            "title": "Spatial Location",
            "description": "Spatial description (e.g., 'top-left', 'center')"
          },
          "hover_explanation": {
            "type": "string",
            "title": "Hover Explanation",
            "description": "Explanation shown on hover"
          },
          "click_explanation": {
            "type": "string",
            "title": "Click Explanation",
            "description": "Detailed explanation shown on click"
          }
        },
        "type": "object",
        "required": [
          "region_id",
          "bounding_box",
          "importance_score",
          "spatial_location",
          "hover_explanation",
          "click_explanation"
        ],
        "title": "InteractiveRegionResponse",
        "description": "Response model for interactive saliency regions."
      },
      "InterpretabilityResponse": {
        "properties": {
          "saliency_map_url": {
            "type": "string",
            "title": "Saliency Map Url",
            "description": "URL to saliency map image"
          },
          "overlay_image_url": {
            "type": "string",
            "title": "Overlay Image Url",
            "description": "URL to overlay visualization"
          },
          "explanation_text": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "Explanation Text",
            "description": "Human-readable explanation"
          },
          "importance_regions": {
            "items": {
              "additionalProperties": true,
              "type": "object"
            },
            "type": "array",
            "title": "Importance Regions",
            "description": "List of important regions with bounding boxes"
          }
        },
        "type": "object",
        "required": [
          "saliency_map_url",
          "overlay_image_url"
        ],
        "title": "InterpretabilityResponse",
        "description": "Response model for interpretability results."
      },
      "LoginRequest": {
        "properties": {
          "password": {
            "type": "string",
            "title": "Password"
          },
          "redirect_url": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "Redirect Url"
          }
        },
        "type": "object",
        "required": [
          "password"
        ],
        "title": "LoginRequest",
        "description": "Login request model."
      },
      "LoginResponse": {
        "properties": {
          "success": {
            "type": "boolean",
            "title": "Success"
          },
          "message": {
            "type": "string",
            "title": "Message"
          },
          "session_token": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "Session Token"
          },
          "redirect_url": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "Redirect Url"
          }
        },
        "type": "object",
        "required": [
          "success",
          "message"
        ],
        "title": "LoginResponse",
        "description": "Login response model."
      },
      "MigrationRequest": {
        "properties": {
          "target_revision": {
            "type": "string",
            "title": "Target Revision",
            "default": "head"
          }
        },
        "type": "object",
        "title": "MigrationRequest",
        "description": "Request model for database migration operations"
      },
      "ModelDeploymentRequest": {
        "properties": {
          "model_parameters_path": {
            "type": "string",
            "title": "Model Parameters Path",
            "description": "Path to trained model parameters"
          },
          "age_group_min": {
            "type": "number",
            "maximum": 18.0,
            "minimum": 2.0,
            "title": "Age Group Min"
          },
          "age_group_max": {
            "type": "number",
            "maximum": 18.0,
            "minimum": 2.0,
            "title": "Age Group Max"
          },
          "replace_existing": {
            "type": "boolean",
            "title": "Replace Existing",
            "description": "Whether to replace existing model for age group",
            "default": false
          }
        },
        "type": "object",
        "required": [
          "model_parameters_path",
          "age_group_min",
          "age_group_max"
        ],
        "title": "ModelDeploymentRequest",
        "description": "Request model for deploying trained model parameters."
      },
      "ModelListResponse": {
        "properties": {
          "models": {
            "items": {
              "$ref": "#/components/schemas/AgeGroupModelResponse"
            },
            "type": "array",
            "title": "Models"
          },
          "total_count": {
            "type": "integer",
            "title": "Total Count"
          },
          "active_count": {
            "type": "integer",
            "title": "Active Count"
          },
          "training_count": {
            "type": "integer",
            "title": "Training Count"
          }
        },
        "type": "object",
        "required": [
          "models",
          "total_count",
          "active_count",
          "training_count"
        ],
        "title": "ModelListResponse",
        "description": "Response model for listing age group models."
      },
      "ModelStatus": {
        "type": "string",
        "enum": [
          "training",
          "ready",
          "failed",
          "insufficient_data"
        ],
        "title": "ModelStatus",
        "description": "Enumeration for model training status."
      },
      "ModelStatusResponse": {
        "properties": {
          "total_models": {
            "type": "integer",
            "minimum": 0.0,
            "title": "Total Models"
          },
          "active_models": {
            "type": "integer",
            "minimum": 0.0,
            "title": "Active Models"
          },
          "training_models": {
            "type": "integer",
            "minimum": 0.0,
            "title": "Training Models"
          },
          "failed_models": {
            "type": "integer",
            "minimum": 0.0,
            "title": "Failed Models"
          },
          "total_drawings": {
            "type": "integer",
            "minimum": 0.0,
            "title": "Total Drawings"
          },
          "total_analyses": {
            "type": "integer",
            "minimum": 0.0,
            "title": "Total Analyses"
          },
          "system_status": {
            "type": "string",
            "title": "System Status"
          },
          "last_training": {
            "anyOf": [
              {
                "type": "string",
                "format": "date-time"
              },
              {
                "type": "null"
              }
            ],
            "title": "Last Training"
          }
        },
        "type": "object",
        "required": [
          "total_models",
          "active_models",
          "training_models",
          "failed_models",
          "total_drawings",
          "total_analyses",
          "system_status"
        ],
        "title": "ModelStatusResponse",
        "description": "Response model for model training and system status."
      },
      "ModelTrainingRequest": {
        "properties": {
          "age_min": {
            "type": "number",
            "maximum": 18.0,
            "minimum": 2.0,
            "title": "Age Min"
          },
          "age_max": {
            "type": "number",
            "maximum": 18.0,
            "minimum": 2.0,
            "title": "Age Max"
          },
          "model_type": {
            "$ref": "#/components/schemas/AnalysisMethod",
            "default": "autoencoder"
          },
          "vision_model": {
            "$ref": "#/components/schemas/VisionModel",
            "default": "vit"
          },
          "min_samples": {
            "type": "integer",
            "minimum": 10.0,
            "title": "Min Samples",
            "description": "Minimum samples required for training",
            "default": 50
          }
        },
        "type": "object",
        "required": [
          "age_min",
          "age_max"
        ],
        "title": "ModelTrainingRequest",
        "description": "Request model for training a new age group model."
      },
      "ResourceCostEstimate": {
        "properties": {
          "service_name": {
            "type": "string",
            "title": "Service Name"
          },
          "monthly_cost_usd": {
            "type": "number",
            "title": "Monthly Cost Usd"
          },
          "resource_type": {
            "type": "string",
            "title": "Resource Type"
          },
          "configuration": {
            "additionalProperties": {
              "type": "string"
            },
            "type": "object",
            "title": "Configuration"
          },
          "optimization_applied": {
            "type": "boolean",
            "title": "Optimization Applied",
            "default": false
          }
        },
        "type": "object",
        "required": [
          "service_name",
          "monthly_cost_usd",
          "resource_type",
          "configuration"
        ],
        "title": "ResourceCostEstimate",
        "description": "Cost estimate for AWS resources."
      },
      "SearchRequest": {
        "properties": {
          "query": {
            "type": "string",
            "title": "Query",
            "description": "Search query string"
          },
          "doc_types": {
            "anyOf": [
              {
                "items": {
                  "type": "string"
                },
                "type": "array"
              },
              {
                "type": "null"
              }
            ],
            "title": "Doc Types",
            "description": "Filter by document types"
          },
          "tags": {
            "anyOf": [
              {
                "items": {
                  "type": "string"
                },
                "type": "array"
              },
              {
                "type": "null"
              }
            ],
            "title": "Tags",
            "description": "Filter by tags"
          },
          "limit": {
            "type": "integer",
            "title": "Limit",
            "description": "Maximum number of results",
            "default": 50
          },
          "offset": {
            "type": "integer",
            "title": "Offset",
            "description": "Result offset for pagination",
            "default": 0
          },
          "include_content": {
            "type": "boolean",
            "title": "Include Content",
            "description": "Include content snippets",
            "default": true
          },
          "highlight": {
            "type": "boolean",
            "title": "Highlight",
            "description": "Highlight search terms",
            "default": true
          }
        },
        "type": "object",
        "required": [
          "query"
        ],
        "title": "SearchRequest",
        "description": "Search request model."
      },
      "SearchResponse": {
        "properties": {
          "results": {
            "items": {
              "$ref": "#/components/schemas/SearchResult"
            },
            "type": "array",
            "title": "Results",
            "description": "Search results"
          },
          "total_count": {
            "type": "integer",
            "title": "Total Count",
            "description": "Total number of results"
          },
          "query_time": {
            "type": "number",
            "title": "Query Time",
            "description": "Query execution time in seconds"
          },
          "facets": {
            "additionalProperties": {
              "additionalProperties": {
                "type": "integer"
              },
              "type": "object"
            },
            "type": "object",
            "title": "Facets",
            "description": "Faceted search results"
          },
          "suggestions": {
            "items": {
              "type": "string"
            },
            "type": "array",
            "title": "Suggestions",
            "description": "Search suggestions"
          },
          "query": {
            "type": "string",
            "title": "Query",
            "description": "Original query"
          }
        },
        "type": "object",
        "required": [
          "total_count",
          "query_time",
          "query"
        ],
        "title": "SearchResponse",
        "description": "Search response model."
      },
      "SearchResult": {
        "properties": {
          "id": {
            "type": "string",
            "title": "Id",
            "description": "Document ID"
          },
          "title": {
            "type": "string",
            "title": "Title",
            "description": "Document title"
          },
          "doc_type": {
            "type": "string",
            "title": "Doc Type",
            "description": "Document type"
          },
          "url": {
            "type": "string",
            "title": "Url",
            "description": "Document URL"
          },
          "score": {
            "type": "number",
            "title": "Score",
            "description": "Relevance score"
          },
          "snippet": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "Snippet",
            "description": "Content snippet"
          },
          "highlights": {
            "items": {
              "type": "string"
            },
            "type": "array",
            "title": "Highlights",
            "description": "Highlighted excerpts"
          },
          "tags": {
            "items": {
              "type": "string"
            },
            "type": "array",
            "title": "Tags",
            "description": "Document tags"
          },
          "last_modified": {
            "type": "string",
            "format": "date-time",
            "title": "Last Modified",
            "description": "Last modification time"
          }
        },
        "type": "object",
        "required": [
          "id",
          "title",
          "doc_type",
          "url",
          "score",
          "last_modified"
        ],
        "title": "SearchResult",
        "description": "Search result model."
      },
      "SecurityAuditRequest": {
        "properties": {
          "iam_role_arn": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "Iam Role Arn",
            "description": "IAM role ARN to validate"
          },
          "s3_buckets": {
            "anyOf": [
              {
                "items": {
                  "type": "string"
                },
                "type": "array"
              },
              {
                "type": "null"
              }
            ],
            "title": "S3 Buckets",
            "description": "S3 bucket names to validate"
          },
          "security_group_ids": {
            "anyOf": [
              {
                "items": {
                  "type": "string"
                },
                "type": "array"
              },
              {
                "type": "null"
              }
            ],
            "title": "Security Group Ids",
            "description": "Security group IDs to validate"
          },
          "vpc_id": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "Vpc Id",
            "description": "VPC ID to validate"
          }
        },
        "type": "object",
        "title": "SecurityAuditRequest",
        "description": "Request model for security audit."
      },
      "SecurityAuditResponse": {
        "properties": {
          "overall_compliant": {
            "type": "boolean",
            "title": "Overall Compliant"
          },
          "total_violations": {
            "type": "integer",
            "title": "Total Violations"
          },
          "total_warnings": {
            "type": "integer",
            "title": "Total Warnings"
          },
          "components": {
            "additionalProperties": {
              "$ref": "#/components/schemas/SecurityValidationResponse"
            },
            "type": "object",
            "title": "Components"
          },
          "summary": {
            "additionalProperties": true,
            "type": "object",
            "title": "Summary"
          }
        },
        "type": "object",
        "required": [
          "overall_compliant",
          "total_violations",
          "total_warnings",
          "components",
          "summary"
        ],
        "title": "SecurityAuditResponse",
        "description": "Response model for comprehensive security audit."
      },
      "SecurityValidationResponse": {
        "properties": {
          "is_compliant": {
            "type": "boolean",
            "title": "Is Compliant"
          },
          "violations": {
            "items": {
              "type": "string"
            },
            "type": "array",
            "title": "Violations"
          },
          "warnings": {
            "items": {
              "type": "string"
            },
            "type": "array",
            "title": "Warnings"
          },
          "recommendations": {
            "items": {
              "type": "string"
            },
            "type": "array",
            "title": "Recommendations"
          },
          "details": {
            "additionalProperties": true,
            "type": "object",
            "title": "Details"
          }
        },
        "type": "object",
        "required": [
          "is_compliant",
          "violations",
          "warnings",
          "recommendations",
          "details"
        ],
        "title": "SecurityValidationResponse",
        "description": "Response model for security validation."
      },
      "SessionStatus": {
        "properties": {
          "authenticated": {
            "type": "boolean",
            "title": "Authenticated"
          },
          "session_info": {
            "anyOf": [
              {
                "additionalProperties": true,
                "type": "object"
              },
              {
                "type": "null"
              }
            ],
            "title": "Session Info"
          },
          "expires_in": {
            "anyOf": [
              {
                "type": "integer"
              },
              {
                "type": "null"
              }
            ],
            "title": "Expires In"
          }
        },
        "type": "object",
        "required": [
          "authenticated"
        ],
        "title": "SessionStatus",
        "description": "Session status model."
      },
      "SimplifiedExplanationResponse": {
        "properties": {
          "summary": {
            "type": "string",
            "title": "Summary",
            "description": "Simple, non-technical explanation"
          },
          "key_findings": {
            "items": {
              "type": "string"
            },
            "type": "array",
            "title": "Key Findings",
            "description": "Main points in accessible language"
          },
          "visual_indicators": {
            "items": {
              "additionalProperties": true,
              "type": "object"
            },
            "type": "array",
            "title": "Visual Indicators",
            "description": "Simple visual cues and their meanings"
          },
          "confidence_level": {
            "type": "string",
            "title": "Confidence Level",
            "description": "High/Medium/Low confidence description"
          },
          "age_appropriate_context": {
            "type": "string",
            "title": "Age Appropriate Context",
            "description": "Context appropriate for the age group"
          },
          "recommendations": {
            "items": {
              "type": "string"
            },
            "type": "array",
            "title": "Recommendations",
            "description": "Simple recommendations"
          }
        },
        "type": "object",
        "required": [
          "summary",
          "key_findings",
          "visual_indicators",
          "confidence_level",
          "age_appropriate_context"
        ],
        "title": "SimplifiedExplanationResponse",
        "description": "Response model for simplified, non-technical explanations."
      },
      "SuccessResponse": {
        "properties": {
          "success": {
            "type": "boolean",
            "title": "Success",
            "default": true
          },
          "message": {
            "type": "string",
            "title": "Message",
            "description": "Success message"
          },
          "data": {
            "anyOf": [
              {
                "additionalProperties": true,
                "type": "object"
              },
              {
                "type": "null"
              }
            ],
            "title": "Data",
            "description": "Response data"
          }
        },
        "type": "object",
        "required": [
          "message"
        ],
        "title": "SuccessResponse",
        "description": "Standard success response model."
      },
      "SystemConfigurationResponse": {
        "properties": {
          "vision_model": {
            "$ref": "#/components/schemas/VisionModel"
          },
          "anomaly_detection_method": {
            "$ref": "#/components/schemas/AnalysisMethod"
          },
          "threshold_percentile": {
            "type": "number",
            "maximum": 99.9,
            "minimum": 50.0,
            "title": "Threshold Percentile"
          },
          "age_grouping_strategy": {
            "$ref": "#/components/schemas/AgeGroupingStrategy"
          },
          "min_samples_per_group": {
            "type": "integer",
            "minimum": 10.0,
            "title": "Min Samples Per Group"
          },
          "max_age_group_span": {
            "type": "number",
            "maximum": 16.0,
            "exclusiveMinimum": 0.0,
            "title": "Max Age Group Span"
          }
        },
        "type": "object",
        "required": [
          "vision_model",
          "anomaly_detection_method",
          "threshold_percentile",
          "age_grouping_strategy",
          "min_samples_per_group",
          "max_age_group_span"
        ],
        "title": "SystemConfigurationResponse",
        "description": "Response model for system configuration."
      },
      "ThresholdUpdateRequest": {
        "properties": {
          "threshold": {
            "type": "number",
            "exclusiveMinimum": 0.0,
            "title": "Threshold",
            "description": "New threshold value"
          },
          "percentile": {
            "anyOf": [
              {
                "type": "number",
                "maximum": 99.9,
                "minimum": 50.0
              },
              {
                "type": "null"
              }
            ],
            "title": "Percentile",
            "description": "Percentile for automatic threshold calculation"
          }
        },
        "type": "object",
        "required": [
          "threshold"
        ],
        "title": "ThresholdUpdateRequest",
        "description": "Request model for updating model thresholds."
      },
      "TrainingConfigRequest": {
        "properties": {
          "job_name": {
            "type": "string",
            "title": "Job Name",
            "description": "Unique name for the training job"
          },
          "environment": {
            "$ref": "#/components/schemas/TrainingEnvironment"
          },
          "dataset_folder": {
            "type": "string",
            "title": "Dataset Folder",
            "description": "Path to folder containing drawings"
          },
          "metadata_file": {
            "type": "string",
            "title": "Metadata File",
            "description": "Path to metadata CSV/JSON file"
          },
          "learning_rate": {
            "type": "number",
            "maximum": 1.0,
            "minimum": 1e-06,
            "title": "Learning Rate",
            "default": 0.001
          },
          "batch_size": {
            "type": "integer",
            "maximum": 512.0,
            "minimum": 1.0,
            "title": "Batch Size",
            "default": 32
          },
          "epochs": {
            "type": "integer",
            "maximum": 1000.0,
            "minimum": 1.0,
            "title": "Epochs",
            "default": 100
          },
          "train_split": {
            "type": "number",
            "maximum": 0.9,
            "minimum": 0.1,
            "title": "Train Split",
            "default": 0.7
          },
          "validation_split": {
            "type": "number",
            "maximum": 0.5,
            "minimum": 0.1,
            "title": "Validation Split",
            "default": 0.2
          },
          "test_split": {
            "type": "number",
            "maximum": 0.3,
            "minimum": 0.05,
            "title": "Test Split",
            "default": 0.1
          },
          "instance_type": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "Instance Type",
            "description": "SageMaker instance type",
            "default": "ml.m5.large"
          },
          "instance_count": {
            "type": "integer",
            "maximum": 10.0,
            "minimum": 1.0,
            "title": "Instance Count",
            "default": 1
          }
        },
        "type": "object",
        "required": [
          "job_name",
          "environment",
          "dataset_folder",
          "metadata_file"
        ],
        "title": "TrainingConfigRequest",
        "description": "Request model for training job configuration."
      },
      "TrainingEnvironment": {
        "type": "string",
        "enum": [
          "local",
          "sagemaker"
        ],
        "title": "TrainingEnvironment",
        "description": "Enumeration for training environments."
      },
      "TrainingJobResponse": {
        "properties": {
          "id": {
            "type": "integer",
            "title": "Id"
          },
          "job_name": {
            "type": "string",
            "title": "Job Name"
          },
          "environment": {
            "type": "string",
            "title": "Environment"
          },
          "status": {
            "type": "string",
            "title": "Status"
          },
          "start_timestamp": {
            "anyOf": [
              {
                "type": "string",
                "format": "date-time"
              },
              {
                "type": "null"
              }
            ],
            "title": "Start Timestamp"
          },
          "end_timestamp": {
            "anyOf": [
              {
                "type": "string",
                "format": "date-time"
              },
              {
                "type": "null"
              }
            ],
            "title": "End Timestamp"
          },
          "sagemaker_job_arn": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "Sagemaker Job Arn"
          }
        },
        "type": "object",
        "required": [
          "id",
          "job_name",
          "environment",
          "status",
          "start_timestamp",
          "end_timestamp",
          "sagemaker_job_arn"
        ],
        "title": "TrainingJobResponse",
        "description": "Response model for training job information."
      },
      "TrainingReportResponse": {
        "properties": {
          "id": {
            "type": "integer",
            "title": "Id"
          },
          "final_loss": {
            "type": "number",
            "title": "Final Loss"
          },
          "validation_accuracy": {
            "type": "number",
            "title": "Validation Accuracy"
          },
          "best_epoch": {
            "type": "integer",
            "title": "Best Epoch"
          },
          "training_time_seconds": {
            "type": "number",
            "title": "Training Time Seconds"
          },
          "model_parameters_path": {
            "type": "string",
            "title": "Model Parameters Path"
          },
          "report_file_path": {
            "type": "string",
            "title": "Report File Path"
          },
          "created_timestamp": {
            "type": "string",
            "format": "date-time",
            "title": "Created Timestamp"
          }
        },
        "type": "object",
        "required": [
          "id",
          "final_loss",
          "validation_accuracy",
          "best_epoch",
          "training_time_seconds",
          "model_parameters_path",
          "report_file_path",
          "created_timestamp"
        ],
        "title": "TrainingReportResponse",
        "description": "Response model for training report information."
      },
      "ValidationError": {
        "properties": {
          "loc": {
            "items": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "integer"
                }
              ]
            },
            "type": "array",
            "title": "Location"
          },
          "msg": {
            "type": "string",
            "title": "Message"
          },
          "type": {
            "type": "string",
            "title": "Error Type"
          }
        },
        "type": "object",
        "required": [
          "loc",
          "msg",
          "type"
        ],
        "title": "ValidationError"
      },
      "VisionModel": {
        "type": "string",
        "enum": [
          "vit"
        ],
        "title": "VisionModel",
        "description": "Enumeration for vision models."
      }
    }
  }
};
    storeOriginalSpec(spec);
    
    // Initialize Swagger UI with enhanced configuration
    window.swaggerUI = SwaggerUIBundle({
        url: './openapi.json',
        dom_id: '#swagger-ui',
        deepLinking: true,
        presets: [
            SwaggerUIBundle.presets.apis,
            SwaggerUIStandalonePreset
        ],
        plugins: [
            SwaggerUIBundle.plugins.DownloadUrl
        ],
        layout: "StandaloneLayout",
        
        // Enhanced configuration options
        displayOperationId: true,
        displayRequestDuration: true,
        docExpansion: "list",
        filter: true,
        showExtensions: true,
        showCommonExtensions: true,
        maxDisplayedTags: 50,
        showMutatedRequest: true,
        supportedSubmitMethods: ["get", "put", "post", "delete", "options", "head", "patch", "trace"],
        
        // Default model expansion
        defaultModelsExpandDepth: 1,
        defaultModelExpandDepth: 1,
        
        // Validator URL (set to null to disable validation)
        validatorUrl: null,
        
        // Try it out enabled by default
        tryItOutEnabled: true,
        
        // Request interceptor for authentication
        requestInterceptor: function(request) {
            // Add custom headers or modify requests here
            console.log('Request interceptor:', request);
            return request;
        },
        
        // Response interceptor for handling responses
        responseInterceptor: function(response) {
            // Handle responses here
            console.log('Response interceptor:', response);
            return response;
        },
        
        // Error handling
        onComplete: function() {
            console.log('Swagger UI loaded successfully');
            
            // Initialize enhancements after UI is ready
            setTimeout(() => {
                setupAdvancedFeatures();
            }, 1000);
        },
        
        onFailure: function(error) {
            console.error('Swagger UI failed to load:', error);
            showNotification('Failed to load API documentation', 'error');
        }
    });
    
    // Store reference globally
    window.swaggerUI = window.swaggerUI;
}

// Configuration constants
const SWAGGER_CONFIG = {
    SEARCH_DEBOUNCE_MS: 300,
    NOTIFICATION_DURATION_MS: 3000,
    COPY_BUTTON_RESET_MS: 2000,
    ENHANCEMENT_INIT_DELAY_MS: 1000,
    
    // Feature flags
    ENABLE_KEYBOARD_SHORTCUTS: true,
    ENABLE_COPY_BUTTONS: true,
    ENABLE_EXPORT_FUNCTIONALITY: true,
    ENABLE_EXPAND_COLLAPSE: true,
    
    // Styling
    PRIMARY_COLOR: '#667eea',
    SUCCESS_COLOR: '#28a745',
    ERROR_COLOR: '#dc3545',
    WARNING_COLOR: '#ffc107'
};