#!/usr/bin/env python3
"""
Direct database script to analyze all drawings without using the API.
This approach directly uses the database and services.
"""

import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent))

from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.database import Drawing, AnomalyAnalysis
from app.api.api_v1.endpoints.analysis import perform_single_analysis
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def analyze_all_drawings_direct(force_reanalysis: bool = False):
    """
    Analyze all drawings directly using the database.
    """
    # Get database session
    db_gen = get_db()
    db: Session = next(db_gen)
    
    try:
        # Get all drawings
        drawings = db.query(Drawing).all()
        logger.info(f"Found {len(drawings)} drawings to analyze")
        
        if not drawings:
            print("No drawings found in database")
            return
        
        successful = 0
        failed = 0
        skipped = 0
        
        for i, drawing in enumerate(drawings, 1):
            print(f"Processing drawing {i}/{len(drawings)}: {drawing.filename}")
            
            # Check if analysis already exists
            if not force_reanalysis:
                existing_analysis = db.query(AnomalyAnalysis).filter(
                    AnomalyAnalysis.drawing_id == drawing.id
                ).first()
                
                if existing_analysis:
                    print(f"  Skipping (already analyzed)")
                    skipped += 1
                    continue
            
            try:
                # Perform analysis
                result = await perform_single_analysis(drawing.id, db, force_reanalysis)
                print(f"  ✓ Analysis complete - Score: {result.analysis.anomaly_score:.4f}, Anomaly: {result.analysis.is_anomaly}")
                successful += 1
                
            except Exception as e:
                print(f"  ✗ Failed: {str(e)}")
                failed += 1
                logger.error(f"Failed to analyze drawing {drawing.id}: {str(e)}")
        
        print(f"\n=== Analysis Complete ===")
        print(f"Total drawings: {len(drawings)}")
        print(f"Successfully analyzed: {successful}")
        print(f"Failed: {failed}")
        print(f"Skipped (already analyzed): {skipped}")
        
    finally:
        db.close()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze all drawings directly via database")
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force re-analysis of drawings that already have results"
    )
    
    args = parser.parse_args()
    
    print("Starting direct analysis of all drawings...")
    print(f"Force re-analysis: {args.force}")
    
    # Run the async function
    asyncio.run(analyze_all_drawings_direct(force_reanalysis=args.force))


if __name__ == "__main__":
    main()