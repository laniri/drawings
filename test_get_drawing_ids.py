#!/usr/bin/env python3
"""
Test the get_all_drawing_ids function.
"""

import requests
from typing import List

API_BASE_URL = "http://localhost:8000/api/v1"

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

if __name__ == "__main__":
    drawing_ids = get_all_drawing_ids()
    print(f"\nFirst 10 drawing IDs: {drawing_ids[:10]}")
    print(f"Last 10 drawing IDs: {drawing_ids[-10:]}")