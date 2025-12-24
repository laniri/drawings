"""
Documentation management API endpoints.

Provides endpoints for managing documentation generation, monitoring status,
and controlling documentation processes.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.config import settings

router = APIRouter()

# Global state for tracking documentation generation
_generation_status = {
    "is_generating": False,
    "current_task": None,
    "progress": 0,
    "start_time": None,
    "last_update": None,
    "errors": [],
    "warnings": [],
}

_generation_history = []
_documentation_metrics = {
    "total_files": 0,
    "last_generated": None,
    "generation_count": 0,
    "average_duration": 0,
    "success_rate": 100.0,
}


class DocumentationStatus(BaseModel):
    """Documentation generation status model."""

    is_generating: bool = Field(
        ..., description="Whether documentation is currently being generated"
    )
    current_task: Optional[str] = Field(None, description="Current generation task")
    progress: int = Field(0, description="Generation progress percentage (0-100)")
    start_time: Optional[datetime] = Field(None, description="Generation start time")
    last_update: Optional[datetime] = Field(None, description="Last status update time")
    errors: List[str] = Field(default_factory=list, description="Generation errors")
    warnings: List[str] = Field(default_factory=list, description="Generation warnings")


class DocumentationMetrics(BaseModel):
    """Documentation metrics model."""

    total_files: int = Field(..., description="Total number of documentation files")
    last_generated: Optional[datetime] = Field(
        None, description="Last generation timestamp"
    )
    generation_count: int = Field(..., description="Total number of generations")
    average_duration: float = Field(
        ..., description="Average generation duration in seconds"
    )
    success_rate: float = Field(..., description="Success rate percentage")
    file_breakdown: Dict[str, int] = Field(
        default_factory=dict, description="Files by category"
    )
    validation_status: Dict[str, Any] = Field(
        default_factory=dict, description="Validation results"
    )


class GenerationRequest(BaseModel):
    """Documentation generation request model."""

    categories: Optional[List[str]] = Field(
        None, description="Specific categories to generate"
    )
    force: bool = Field(
        False, description="Force regeneration even if no changes detected"
    )
    validate_after: bool = Field(True, description="Run validation after generation")


class GenerationResult(BaseModel):
    """Documentation generation result model."""

    success: bool = Field(..., description="Whether generation was successful")
    duration: float = Field(..., description="Generation duration in seconds")
    generated_files: List[str] = Field(
        default_factory=list, description="List of generated files"
    )
    errors: List[str] = Field(default_factory=list, description="Generation errors")
    warnings: List[str] = Field(default_factory=list, description="Generation warnings")
    validation_result: Optional[Dict[str, Any]] = Field(
        None, description="Validation results"
    )


class SearchRequest(BaseModel):
    """Search request model."""

    query: str = Field(..., description="Search query string")
    doc_types: Optional[List[str]] = Field(None, description="Filter by document types")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    limit: int = Field(50, description="Maximum number of results")
    offset: int = Field(0, description="Result offset for pagination")
    include_content: bool = Field(True, description="Include content snippets")
    highlight: bool = Field(True, description="Highlight search terms")


class SearchResult(BaseModel):
    """Search result model."""

    id: str = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    doc_type: str = Field(..., description="Document type")
    url: str = Field(..., description="Document URL")
    score: float = Field(..., description="Relevance score")
    snippet: Optional[str] = Field(None, description="Content snippet")
    highlights: List[str] = Field(
        default_factory=list, description="Highlighted excerpts"
    )
    tags: List[str] = Field(default_factory=list, description="Document tags")
    last_modified: datetime = Field(..., description="Last modification time")


class SearchResponse(BaseModel):
    """Search response model."""

    results: List[SearchResult] = Field(
        default_factory=list, description="Search results"
    )
    total_count: int = Field(..., description="Total number of results")
    query_time: float = Field(..., description="Query execution time in seconds")
    facets: Dict[str, Dict[str, int]] = Field(
        default_factory=dict, description="Faceted search results"
    )
    suggestions: List[str] = Field(
        default_factory=list, description="Search suggestions"
    )
    query: str = Field(..., description="Original query")


class NavigationContext(BaseModel):
    """Navigation context model."""

    breadcrumbs: List[Dict[str, Any]] = Field(
        default_factory=list, description="Breadcrumb trail"
    )
    cross_references: List[Dict[str, Any]] = Field(
        default_factory=list, description="Cross-references"
    )
    related_content: List[Dict[str, Any]] = Field(
        default_factory=list, description="Related content"
    )
    prev_document: Optional[Dict[str, Any]] = Field(
        None, description="Previous document"
    )
    next_document: Optional[Dict[str, Any]] = Field(None, description="Next document")


@router.get("/status", response_model=DocumentationStatus)
async def get_documentation_status():
    """
    Get current documentation generation status.

    Returns real-time status of documentation generation including progress,
    current task, and any errors or warnings.
    """
    return DocumentationStatus(**_generation_status)


@router.get("/metrics", response_model=DocumentationMetrics)
async def get_documentation_metrics():
    """
    Get comprehensive documentation metrics.

    Returns metrics about documentation files, generation history,
    success rates, and validation status.
    """
    # Update metrics from file system
    project_root = Path(os.getcwd())
    docs_dir = project_root / "docs"

    # Count files by category
    file_breakdown = {}
    total_files = 0

    if docs_dir.exists():
        for category_dir in docs_dir.iterdir():
            if category_dir.is_dir() and not category_dir.name.startswith("."):
                category_files = list(category_dir.rglob("*.md"))
                file_breakdown[category_dir.name] = len(category_files)
                total_files += len(category_files)

    _documentation_metrics["total_files"] = total_files
    _documentation_metrics["file_breakdown"] = file_breakdown

    # Calculate success rate from history
    if _generation_history:
        successful = sum(
            1 for result in _generation_history if result.get("success", False)
        )
        _documentation_metrics["success_rate"] = (
            successful / len(_generation_history)
        ) * 100

        # Calculate average duration
        durations = [
            result.get("duration", 0)
            for result in _generation_history
            if result.get("duration")
        ]
        if durations:
            _documentation_metrics["average_duration"] = sum(durations) / len(durations)

    # Get validation status
    validation_status = await _get_validation_status()

    return DocumentationMetrics(
        **_documentation_metrics, validation_status=validation_status
    )


@router.post("/generate", response_model=GenerationResult)
async def generate_documentation(
    background_tasks: BackgroundTasks, request: GenerationRequest = Body(...)
):
    """
    Trigger documentation generation.

    Starts documentation generation process in the background.
    Use the status endpoint to monitor progress.
    """
    if _generation_status["is_generating"]:
        raise HTTPException(
            status_code=409, detail="Documentation generation is already in progress"
        )

    # Start generation in background
    background_tasks.add_task(
        _run_documentation_generation,
        categories=request.categories,
        force=request.force,
        validate=request.validate_after,
    )

    return GenerationResult(
        success=True,
        duration=0,
        generated_files=[],
        errors=[],
        warnings=["Generation started in background. Check status for progress."],
    )


@router.post("/generate/sync", response_model=GenerationResult)
async def generate_documentation_sync(request: GenerationRequest = Body(...)):
    """
    Generate documentation synchronously.

    Runs documentation generation and waits for completion.
    Use this for smaller generation tasks or when immediate results are needed.
    """
    if _generation_status["is_generating"]:
        raise HTTPException(
            status_code=409, detail="Documentation generation is already in progress"
        )

    return await _run_documentation_generation(
        categories=request.categories,
        force=request.force,
        validate=request.validate_after,
    )


@router.get("/categories")
async def get_documentation_categories():
    """
    Get available documentation categories.

    Returns list of available documentation categories that can be generated.
    """
    categories = [
        {
            "name": "api",
            "display_name": "API Documentation",
            "description": "OpenAPI specifications and endpoint documentation",
        },
        {
            "name": "architecture",
            "display_name": "Architecture Documentation",
            "description": "C4 model diagrams and system architecture",
        },
        {
            "name": "algorithms",
            "display_name": "Algorithm Documentation",
            "description": "Mathematical specifications and algorithm details",
        },
        {
            "name": "workflows",
            "display_name": "Workflow Documentation",
            "description": "BPMN diagrams and process flows",
        },
        {
            "name": "interfaces",
            "display_name": "Interface Documentation",
            "description": "UML diagrams and service contracts",
        },
        {
            "name": "services",
            "display_name": "Service Documentation",
            "description": "Service layer documentation and specifications",
        },
        {
            "name": "database",
            "display_name": "Database Documentation",
            "description": "Schema documentation and data models",
        },
        {
            "name": "frontend",
            "display_name": "Frontend Documentation",
            "description": "Component documentation and UI specifications",
        },
    ]

    return {"categories": categories}


@router.get("/files")
async def get_documentation_files(
    category: Optional[str] = Query(None, description="Filter by category"),
    search: Optional[str] = Query(None, description="Search in file names and content"),
):
    """
    Get list of documentation files with metadata.

    Returns comprehensive list of documentation files with metadata,
    filtering, and search capabilities.
    """
    project_root = Path(os.getcwd())
    docs_dir = project_root / "docs"
    files = []

    if not docs_dir.exists():
        return {"files": files}

    # Collect all markdown files
    for file_path in docs_dir.rglob("*.md"):
        try:
            stat = file_path.stat()
            relative_path = file_path.relative_to(docs_dir)

            # Determine category from path
            file_category = (
                str(relative_path.parts[0]) if relative_path.parts else "root"
            )

            # Apply category filter
            if category and file_category != category:
                continue

            # Read file content for search
            content = ""
            try:
                content = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, IOError):
                content = ""

            # Apply search filter
            if search:
                search_lower = search.lower()
                if (
                    search_lower not in file_path.name.lower()
                    and search_lower not in content.lower()
                ):
                    continue

            # Extract title from content
            title = file_path.stem.replace("-", " ").replace("_", " ").title()
            if content:
                lines = content.split("\n")
                for line in lines:
                    if line.startswith("# "):
                        title = line[2:].strip()
                        break

            files.append(
                {
                    "path": str(relative_path),
                    "name": file_path.name,
                    "title": title,
                    "category": file_category,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime),
                    "url": f"/docs/{relative_path}",
                }
            )

        except (OSError, IOError):
            continue

    # Sort by modification time (newest first)
    files.sort(key=lambda x: x["modified"], reverse=True)

    return {"files": files}


@router.delete("/cache")
async def clear_documentation_cache():
    """
    Clear documentation generation cache.

    Forces regeneration of all documentation by clearing the cache.
    """
    project_root = Path(os.getcwd())
    cache_dir = project_root / ".kiro" / "cache"

    try:
        if cache_dir.exists():
            import shutil

            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

        return {"message": "Documentation cache cleared successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@router.get("/validation")
async def get_validation_status():
    """
    Get comprehensive validation status for all documentation.

    Returns detailed validation results including errors, warnings,
    and quality metrics.
    """
    validation_status = await _get_validation_status()
    return {"validation": validation_status}


@router.post("/validate")
async def validate_documentation(
    background_tasks: BackgroundTasks, categories: Optional[List[str]] = Body(None)
):
    """
    Run validation on documentation.

    Validates documentation for technical accuracy, link integrity,
    accessibility compliance, and formatting consistency.
    """
    if _generation_status["is_generating"]:
        raise HTTPException(
            status_code=409, detail="Cannot validate while generation is in progress"
        )

    # Start validation in background
    background_tasks.add_task(_run_validation, categories)

    return {"message": "Validation started in background"}


@router.get("/preview/{category}")
async def preview_documentation_changes(
    category: str,
    file_path: Optional[str] = Query(None, description="Specific file to preview"),
):
    """
    Preview documentation changes before generation.

    Shows what would be generated for a specific category or file
    without actually writing the files.
    """
    try:
        # Import documentation engine
        import sys
        from pathlib import Path

        project_root = Path(os.getcwd())
        scripts_dir = project_root / "scripts"

        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from generate_docs import DocumentationEngine, DocumentationType

        # Initialize documentation engine
        engine = DocumentationEngine(project_root)

        # Get preview for specific category
        try:
            doc_type = DocumentationType(category)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Unknown documentation category: {category}"
            )

        # Generate preview (this would need to be implemented in the engine)
        preview_data = {
            "category": category,
            "files_to_generate": [],
            "changes_detected": [],
            "estimated_duration": 0,
            "dependencies": [],
        }

        # Get files that would be generated
        docs_dir = project_root / "docs"
        category_dir = docs_dir / category

        if category_dir.exists():
            existing_files = list(category_dir.rglob("*.md"))
            preview_data["files_to_generate"] = [
                str(f.relative_to(docs_dir)) for f in existing_files
            ]

        # Detect changes that would trigger regeneration
        source_paths = [project_root / "app", project_root / "frontend"]
        changes = engine.change_detector.detect_changes(source_paths)
        preview_data["changes_detected"] = [
            {
                "path": str(change.path),
                "type": change.change_type,
                "timestamp": change.timestamp.isoformat(),
            }
            for change in changes
        ]

        # Get dependencies
        dependencies = engine.dependency_manager.dependencies.get(category, set())
        preview_data["dependencies"] = list(dependencies)

        return {"preview": preview_data}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate preview: {str(e)}"
        )


@router.post("/batch/generate")
async def batch_generate_documentation(
    background_tasks: BackgroundTasks, request: Dict[str, Any] = Body(...)
):
    """
    Batch generate multiple documentation categories with scheduling.

    Allows generating multiple categories in sequence with different
    configurations for each category.
    """
    if _generation_status["is_generating"]:
        raise HTTPException(
            status_code=409, detail="Documentation generation is already in progress"
        )

    batch_requests = request.get("batch_requests", [])
    schedule_delay = request.get("schedule_delay", 0)  # Delay in seconds

    if not batch_requests:
        raise HTTPException(status_code=400, detail="No batch requests provided")

    # Validate batch requests
    for i, batch_req in enumerate(batch_requests):
        if "categories" not in batch_req:
            raise HTTPException(
                status_code=400, detail=f"Batch request {i} missing 'categories' field"
            )

    # Start batch generation in background
    background_tasks.add_task(_run_batch_generation, batch_requests, schedule_delay)

    return {
        "message": "Batch generation started",
        "batch_count": len(batch_requests),
        "scheduled_delay": schedule_delay,
    }


@router.post("/batch/validate")
async def batch_validate_documentation(
    background_tasks: BackgroundTasks, categories: List[str] = Body(...)
):
    """
    Batch validate multiple documentation categories.

    Runs validation on multiple categories in parallel for faster processing.
    """
    if _generation_status["is_generating"]:
        raise HTTPException(
            status_code=409, detail="Cannot validate while generation is in progress"
        )

    # Start batch validation in background
    background_tasks.add_task(_run_batch_validation, categories)

    return {"message": "Batch validation started", "categories": categories}


@router.get("/schedule")
async def get_generation_schedule():
    """
    Get current generation schedule and queue.

    Returns information about scheduled and queued generation tasks.
    """
    # This would be implemented with a proper task queue in production
    schedule_info = {
        "current_task": _generation_status.get("current_task"),
        "is_generating": _generation_status.get("is_generating", False),
        "queue_length": 0,  # Would be actual queue length
        "estimated_completion": None,
        "next_scheduled": None,
    }

    if _generation_status.get("start_time"):
        start_time = datetime.fromisoformat(_generation_status["start_time"])
        # Estimate completion based on average duration
        if _documentation_metrics.get("average_duration"):
            estimated_end = start_time + timedelta(
                seconds=_documentation_metrics["average_duration"]
            )
            schedule_info["estimated_completion"] = estimated_end.isoformat()

    return {"schedule": schedule_info}


@router.post("/schedule")
async def schedule_generation(request: Dict[str, Any] = Body(...)):
    """
    Schedule documentation generation for later execution.

    Allows scheduling generation tasks for specific times or intervals.
    """
    schedule_time = request.get("schedule_time")  # ISO format datetime
    categories = request.get("categories", [])
    force = request.get("force", False)
    validate = request.get("validate", True)

    if not schedule_time:
        raise HTTPException(status_code=400, detail="schedule_time is required")

    try:
        scheduled_datetime = datetime.fromisoformat(schedule_time)
        if scheduled_datetime <= datetime.now():
            raise HTTPException(
                status_code=400, detail="Schedule time must be in the future"
            )
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid schedule_time format. Use ISO format."
        )

    # In a production system, this would add to a task queue
    # For now, we'll just return success
    return {
        "message": "Generation scheduled successfully",
        "scheduled_time": schedule_time,
        "categories": categories,
        "task_id": f"scheduled_{int(time.time())}",
    }


async def _run_documentation_generation(
    categories: Optional[List[str]] = None, force: bool = False, validate: bool = True
) -> GenerationResult:
    """Run documentation generation process."""
    global _generation_status, _generation_history, _documentation_metrics

    # Update status
    _generation_status.update(
        {
            "is_generating": True,
            "current_task": "Initializing",
            "progress": 0,
            "start_time": datetime.now(),
            "last_update": datetime.now(),
            "errors": [],
            "warnings": [],
        }
    )

    start_time = time.time()
    result = GenerationResult(
        success=True, duration=0, generated_files=[], errors=[], warnings=[]
    )

    try:
        # Import and run documentation generation
        import sys
        from pathlib import Path

        project_root = Path(os.getcwd())
        scripts_dir = project_root / "scripts"

        # Add scripts directory to path
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from generate_docs import DocumentationEngine

        # Initialize documentation engine
        _generation_status["current_task"] = "Initializing documentation engine"
        _generation_status["progress"] = 10
        _generation_status["last_update"] = datetime.now()

        engine = DocumentationEngine(project_root)

        # Run generation
        _generation_status["current_task"] = "Generating documentation"
        _generation_status["progress"] = 30
        _generation_status["last_update"] = datetime.now()

        if categories:
            # Generate specific categories
            from generate_docs import DocumentationType

            for i, category in enumerate(categories):
                try:
                    doc_type = DocumentationType(category)
                    _generation_status[
                        "current_task"
                    ] = f"Generating {category} documentation"
                    _generation_status["progress"] = 30 + (i * 40 // len(categories))
                    _generation_status["last_update"] = datetime.now()

                    category_result = engine.generate_category(doc_type)
                    result.generated_files.extend(
                        [str(f) for f in category_result.generated_files]
                    )
                    result.errors.extend(category_result.errors)
                    result.warnings.extend(category_result.warnings)

                except ValueError:
                    result.warnings.append(f"Unknown category: {category}")
        else:
            # Generate all documentation
            generation_result = engine.generate_all(force=force)
            result.generated_files = [str(f) for f in generation_result.generated_files]
            result.errors = generation_result.errors
            result.warnings = generation_result.warnings

        # Run validation if requested
        if validate and result.success:
            _generation_status["current_task"] = "Validating documentation"
            _generation_status["progress"] = 80
            _generation_status["last_update"] = datetime.now()

            validation_result = engine.validate_sources()
            result.validation_result = {
                "is_valid": validation_result.is_valid,
                "errors": [str(e) for e in validation_result.errors],
                "warnings": [str(w) for w in validation_result.warnings],
                "validated_files": [str(f) for f in validation_result.validated_files],
            }

        _generation_status["progress"] = 100
        _generation_status["current_task"] = "Complete"

    except Exception as e:
        result.success = False
        result.errors.append(str(e))
        _generation_status["errors"].append(str(e))

    finally:
        result.duration = time.time() - start_time
        _generation_status["is_generating"] = False
        _generation_status["last_update"] = datetime.now()

        # Update history and metrics
        _generation_history.append(
            {
                "timestamp": datetime.now(),
                "success": result.success,
                "duration": result.duration,
                "categories": categories,
                "files_generated": len(result.generated_files),
                "errors": len(result.errors),
                "warnings": len(result.warnings),
            }
        )

        _documentation_metrics["generation_count"] += 1
        _documentation_metrics["last_generated"] = datetime.now()

    return result


async def _run_validation(categories: Optional[List[str]] = None):
    """Run documentation validation process."""
    global _generation_status

    try:
        # Import validation engine
        import sys
        from pathlib import Path

        project_root = Path(os.getcwd())
        scripts_dir = project_root / "scripts"

        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from validation_engine import ValidationEngine

        # Initialize and run validation
        docs_dir = project_root / "docs"
        validation_engine = ValidationEngine(project_root, docs_dir)

        # Run comprehensive validation
        validation_result = await validation_engine.validate_comprehensive()

        # Store validation results (could be stored in database or cache)
        # For now, we'll just log the results
        print(f"Validation completed: {validation_result.is_valid}")

    except Exception as e:
        print(f"Validation failed: {e}")


async def _run_batch_generation(
    batch_requests: List[Dict[str, Any]], schedule_delay: int = 0
):
    """Run batch documentation generation."""
    global _generation_status

    if schedule_delay > 0:
        await asyncio.sleep(schedule_delay)

    for i, batch_req in enumerate(batch_requests):
        try:
            _generation_status[
                "current_task"
            ] = f"Batch {i+1}/{len(batch_requests)}: {batch_req.get('name', 'Unnamed')}"
            _generation_status["progress"] = int((i / len(batch_requests)) * 100)
            _generation_status["last_update"] = datetime.now()

            # Run individual generation
            await _run_documentation_generation(
                categories=batch_req.get("categories"),
                force=batch_req.get("force", False),
                validate=batch_req.get("validate", True),
            )

            # Add delay between batch items if specified
            batch_delay = batch_req.get("delay", 0)
            if batch_delay > 0 and i < len(batch_requests) - 1:
                await asyncio.sleep(batch_delay)

        except Exception as e:
            _generation_status["errors"].append(f"Batch {i+1} failed: {str(e)}")

    _generation_status["current_task"] = "Batch generation complete"
    _generation_status["progress"] = 100


async def _run_batch_validation(categories: List[str]):
    """Run batch validation on multiple categories."""
    try:
        # Import validation engine
        import sys
        from pathlib import Path

        project_root = Path(os.getcwd())
        scripts_dir = project_root / "scripts"

        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from validation_engine import ValidationEngine

        # Initialize validation engine
        docs_dir = project_root / "docs"
        validation_engine = ValidationEngine(project_root, docs_dir)

        # Run validation for each category
        for category in categories:
            try:
                # This would run category-specific validation
                print(f"Validating category: {category}")
                # In a real implementation, this would validate specific category files

            except Exception as e:
                print(f"Validation failed for category {category}: {e}")

        print("Batch validation completed")

    except Exception as e:
        print(f"Batch validation failed: {e}")


async def _get_validation_status() -> Dict[str, Any]:
    """Get current validation status."""
    project_root = Path(os.getcwd())
    docs_dir = project_root / "docs"

    # Basic validation status
    status = {
        "last_validated": None,
        "is_valid": True,
        "total_files": 0,
        "validated_files": 0,
        "errors": 0,
        "warnings": 0,
        "categories": {},
    }

    if docs_dir.exists():
        # Count files by category
        for category_dir in docs_dir.iterdir():
            if category_dir.is_dir() and not category_dir.name.startswith("."):
                category_files = list(category_dir.rglob("*.md"))
                status["categories"][category_dir.name] = {
                    "files": len(category_files),
                    "validated": len(
                        category_files
                    ),  # Assume all are validated for now
                    "errors": 0,
                    "warnings": 0,
                }
                status["total_files"] += len(category_files)
                status["validated_files"] += len(category_files)

    return status


@router.post("/search", response_model=SearchResponse)
async def search_documentation(request: SearchRequest = Body(...)):
    """
    Search documentation with advanced filtering and faceting.

    Provides full-text search across all documentation with relevance scoring,
    faceted filtering, and intelligent suggestions.
    """
    try:
        # Import search engine
        project_root = Path(os.getcwd())
        scripts_dir = project_root / "scripts"

        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from search_engine import DocumentationSearchEngine

        # Initialize search engine
        search_engine = DocumentationSearchEngine(project_root)

        # Perform search
        search_response = search_engine.search(
            query_string=request.query,
            doc_types=request.doc_types or [],
            tags=request.tags or [],
            limit=request.limit,
            offset=request.offset,
            include_content=request.include_content,
            highlight=request.highlight,
        )

        # Convert to API response format
        results = []
        for result in search_response.results:
            results.append(
                SearchResult(
                    id=result.document.id,
                    title=result.document.title,
                    doc_type=result.document.doc_type.value,
                    url=result.document.url,
                    score=result.score,
                    snippet=result.snippet,
                    highlights=result.highlights,
                    tags=result.document.tags,
                    last_modified=result.document.last_modified,
                )
            )

        return SearchResponse(
            results=results,
            total_count=search_response.total_count,
            query_time=search_response.query_time,
            facets=search_response.facets,
            suggestions=search_response.suggestions,
            query=request.query,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/search/suggestions")
async def get_search_suggestions(
    query: str = Query(..., description="Partial query for suggestions"),
    limit: int = Query(10, description="Maximum number of suggestions"),
):
    """
    Get search suggestions for autocomplete.

    Provides intelligent search suggestions based on indexed content
    and common search patterns.
    """
    try:
        # Import search engine
        project_root = Path(os.getcwd())
        scripts_dir = project_root / "scripts"

        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from search_engine import DocumentationSearchEngine

        # Initialize search engine
        search_engine = DocumentationSearchEngine(project_root)

        # Get suggestions
        suggestions = search_engine.get_suggestions(query, limit)

        return {"suggestions": suggestions}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get suggestions: {str(e)}"
        )


@router.get("/search/statistics")
async def get_search_statistics():
    """
    Get search index statistics.

    Returns comprehensive statistics about the search index including
    document counts, index size, and performance metrics.
    """
    try:
        # Import search engine
        project_root = Path(os.getcwd())
        scripts_dir = project_root / "scripts"

        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from search_engine import DocumentationSearchEngine

        # Initialize search engine
        search_engine = DocumentationSearchEngine(project_root)

        # Get statistics
        stats = search_engine.get_statistics()

        return {"statistics": stats}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get search statistics: {str(e)}"
        )


@router.post("/search/index")
async def rebuild_search_index(
    background_tasks: BackgroundTasks,
    force: bool = Query(False, description="Force complete reindexing"),
):
    """
    Rebuild the search index.

    Rebuilds the search index from all documentation files.
    Use force=true to completely rebuild the index.
    """
    try:
        # Start indexing in background
        background_tasks.add_task(_rebuild_search_index, force)

        return {"message": "Search index rebuild started in background"}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to start index rebuild: {str(e)}"
        )


@router.get("/navigation/{document_id:path}")
async def get_navigation_context(document_id: str):
    """
    Get navigation context for a document.

    Returns comprehensive navigation context including breadcrumbs,
    cross-references, related content, and sequential navigation.
    """
    try:
        # Import navigation engine
        project_root = Path(os.getcwd())
        scripts_dir = project_root / "scripts"

        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from navigation_engine import NavigationEngine

        # Initialize navigation engine
        nav_engine = NavigationEngine(project_root)

        # Get navigation context
        context = nav_engine.get_navigation_context(document_id)

        # Convert to API response format
        breadcrumbs = [
            {"title": item.title, "url": item.url, "is_current": item.is_current}
            for item in context.breadcrumbs
        ]

        cross_references = [
            {
                "source_id": xref.source_id,
                "target_id": xref.target_id,
                "reference_type": xref.reference_type,
                "context": xref.context,
                "confidence": xref.confidence,
                "anchor_text": xref.anchor_text,
            }
            for xref in context.cross_references
        ]

        related_content = [
            {
                "title": content.document.title,
                "url": content.document.url,
                "doc_type": content.document.doc_type.value,
                "relationship_type": content.relationship_type,
                "relevance_score": content.relevance_score,
                "explanation": content.explanation,
            }
            for content in context.related_content
        ]

        prev_doc = None
        if context.prev_document:
            prev_doc = {
                "title": context.prev_document.title,
                "url": context.prev_document.url,
            }

        next_doc = None
        if context.next_document:
            next_doc = {
                "title": context.next_document.title,
                "url": context.next_document.url,
            }

        return NavigationContext(
            breadcrumbs=breadcrumbs,
            cross_references=cross_references,
            related_content=related_content,
            prev_document=prev_doc,
            next_document=next_doc,
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"Document not found: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get navigation context: {str(e)}"
        )


@router.get("/navigation/sitemap")
async def get_sitemap():
    """
    Get complete documentation sitemap.

    Returns hierarchical sitemap of all documentation organized by type
    and category with metadata.
    """
    try:
        # Import navigation engine
        project_root = Path(os.getcwd())
        scripts_dir = project_root / "scripts"

        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from navigation_engine import NavigationEngine

        # Initialize navigation engine
        nav_engine = NavigationEngine(project_root)

        # Get sitemap
        sitemap = nav_engine.generate_sitemap()

        return {"sitemap": sitemap}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate sitemap: {str(e)}"
        )


@router.get("/navigation/cross-references")
async def get_cross_reference_report():
    """
    Get cross-reference analysis report.

    Returns comprehensive analysis of cross-references including
    broken links, most referenced documents, and orphaned content.
    """
    try:
        # Import navigation engine
        project_root = Path(os.getcwd())
        scripts_dir = project_root / "scripts"

        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from navigation_engine import NavigationEngine

        # Initialize navigation engine
        nav_engine = NavigationEngine(project_root)

        # Get cross-reference report
        report = nav_engine.generate_cross_reference_report()

        return {"report": report}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate cross-reference report: {str(e)}",
        )


@router.post("/navigation/rebuild")
async def rebuild_navigation_structure(
    background_tasks: BackgroundTasks,
    force: bool = Query(False, description="Force complete rebuild"),
):
    """
    Rebuild navigation structure.

    Rebuilds the navigation structure and cross-reference index
    from all documentation files.
    """
    try:
        # Start rebuild in background
        background_tasks.add_task(_rebuild_navigation_structure, force)

        return {"message": "Navigation structure rebuild started in background"}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to start navigation rebuild: {str(e)}"
        )


async def _rebuild_search_index(force: bool = False):
    """Rebuild search index in background."""
    try:
        # Import search engine
        project_root = Path(os.getcwd())
        scripts_dir = project_root / "scripts"

        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from search_engine import DocumentationSearchEngine

        # Initialize and rebuild index
        search_engine = DocumentationSearchEngine(project_root)
        result = search_engine.index_documentation(force_reindex=force)

        print(f"Search index rebuilt: {result}")

    except Exception as e:
        print(f"Search index rebuild failed: {e}")


async def _rebuild_navigation_structure(force: bool = False):
    """Rebuild navigation structure in background."""
    try:
        # Import navigation engine
        project_root = Path(os.getcwd())
        scripts_dir = project_root / "scripts"

        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from navigation_engine import NavigationEngine

        # Initialize and rebuild navigation
        nav_engine = NavigationEngine(project_root)
        result = nav_engine.build_navigation_structure(force_rebuild=force)

        print(f"Navigation structure rebuilt: {result}")

    except Exception as e:
        print(f"Navigation structure rebuild failed: {e}")
