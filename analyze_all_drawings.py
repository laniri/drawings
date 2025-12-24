#!/usr/bin/env python3
"""
Script to analyze all drawings in the database using the batch analysis API.
"""

import requests
import time
import json
import sys
from typing import List, Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
BATCH_SIZE = 50  # Process drawings in batches to avoid overwhelming the system


def get_all_drawing_ids() -> List[int]:
    """Get all drawing IDs from the database."""
    try:
        # Start with first page using maximum allowed page size
        page_size = 100  # API maximum
        page = 1
        all_drawing_ids = []
        
        print("Fetching drawing IDs from API...")
        
        while True:
            response = requests.get(f"{API_BASE_URL}/drawings/?page={page}&page_size={page_size}")
            response.raise_for_status()
            data = response.json()
            
            # Handle the DrawingListResponse format
            if isinstance(data, dict) and "drawings" in data:
                drawings = data["drawings"]
                total_pages = data.get("total_pages", 1)
                total_count = data.get("total_count", 0)
            else:
                drawings = data
                total_pages = 1
                total_count = len(drawings)
            
            # Extract IDs from this page
            page_ids = [drawing["id"] for drawing in drawings]
            all_drawing_ids.extend(page_ids)
            
            print(f"  Fetched page {page}/{total_pages} ({len(page_ids)} drawings)")
            
            # Check if we have more pages
            if page >= total_pages or len(drawings) < page_size:
                break
                
            page += 1
        
        print(f"Total drawing IDs fetched: {len(all_drawing_ids)}")
        return all_drawing_ids
        
    except requests.RequestException as e:
        print(f"Error fetching drawings: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return []


def submit_batch_analysis(drawing_ids: List[int], force_reanalysis: bool = False) -> str:
    """Submit a batch analysis request."""
    payload = {
        "drawing_ids": drawing_ids,
        "force_reanalysis": force_reanalysis
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/analysis/batch", json=payload)
        response.raise_for_status()
        result = response.json()
        return result["batch_id"]
    except requests.RequestException as e:
        print(f"Error submitting batch analysis: {e}")
        return None


def check_batch_progress(batch_id: str) -> Dict[str, Any]:
    """Check the progress of a batch analysis."""
    try:
        response = requests.get(f"{API_BASE_URL}/analysis/batch/{batch_id}/progress")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error checking batch progress: {e}")
        return {}


def wait_for_batch_completion(batch_id: str, check_interval: int = 10) -> Dict[str, Any]:
    """Wait for batch analysis to complete."""
    print(f"Waiting for batch {batch_id} to complete...")
    
    while True:
        progress = check_batch_progress(batch_id)
        if not progress:
            print("Failed to get batch progress")
            return {}
        
        status = progress.get("status", "unknown")
        completed = progress.get("completed", 0)
        failed = progress.get("failed", 0)
        total = progress.get("total_drawings", 0)
        
        print(f"Status: {status} | Completed: {completed}/{total} | Failed: {failed}")
        
        if status == "completed":
            print("Batch analysis completed!")
            return progress
        elif status.startswith("failed"):
            print(f"Batch analysis failed: {status}")
            return progress
        
        time.sleep(check_interval)


def analyze_all_drawings(force_reanalysis: bool = False):
    """Analyze all drawings in the database."""
    print("Fetching all drawing IDs...")
    drawing_ids = get_all_drawing_ids()
    
    if not drawing_ids:
        print("No drawings found in database")
        return
    
    print(f"Found {len(drawing_ids)} drawings to analyze")
    
    # Process in batches
    batch_results = []
    for i in range(0, len(drawing_ids), BATCH_SIZE):
        batch_drawings = drawing_ids[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(drawing_ids) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch_drawings)} drawings)")
        
        batch_id = submit_batch_analysis(batch_drawings, force_reanalysis)
        if not batch_id:
            print(f"Failed to submit batch {batch_num}")
            continue
        
        print(f"Submitted batch {batch_id}")
        result = wait_for_batch_completion(batch_id)
        batch_results.append(result)
    
    # Summary
    total_completed = sum(r.get("completed", 0) for r in batch_results)
    total_failed = sum(r.get("failed", 0) for r in batch_results)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Total drawings processed: {len(drawing_ids)}")
    print(f"Successfully analyzed: {total_completed}")
    print(f"Failed: {total_failed}")
    
    if total_failed > 0:
        print("\nFailed analyses:")
        for i, result in enumerate(batch_results):
            errors = result.get("errors", [])
            if errors:
                print(f"Batch {i+1} errors:")
                for error in errors:
                    print(f"  - Drawing {error.get('drawing_id')}: {error.get('error')}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze all drawings in the database")
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force re-analysis of drawings that already have results"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=50, 
        help="Number of drawings to process in each batch (default: 50)"
    )
    
    args = parser.parse_args()
    
    global BATCH_SIZE
    BATCH_SIZE = args.batch_size
    
    print("Starting analysis of all drawings...")
    print(f"Force re-analysis: {args.force}")
    print(f"Batch size: {BATCH_SIZE}")
    
    analyze_all_drawings(force_reanalysis=args.force)


if __name__ == "__main__":
    main()