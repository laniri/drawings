#!/usr/bin/env python3
"""
Check the analysis status of all drawings in the database.
"""

import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent))

from sqlalchemy.orm import Session
from sqlalchemy import func
from app.core.database import get_db
from app.models.database import Drawing, AnomalyAnalysis
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_analysis_status():
    """Check which drawings have been analyzed and which haven't."""
    # Get database session
    db_gen = get_db()
    db: Session = next(db_gen)
    
    try:
        # Get total drawing count
        total_drawings = db.query(Drawing).count()
        
        # Get analyzed drawing count
        analyzed_drawings = db.query(AnomalyAnalysis.drawing_id).distinct().count()
        
        # Get unanalyzed drawings
        analyzed_drawing_ids = db.query(AnomalyAnalysis.drawing_id).distinct().subquery()
        unanalyzed_drawings = db.query(Drawing).filter(
            ~Drawing.id.in_(analyzed_drawing_ids)
        ).all()
        
        # Get age distribution of unanalyzed drawings
        age_distribution = {}
        for drawing in unanalyzed_drawings:
            age_group = f"{int(drawing.age_years)}-{int(drawing.age_years)+1}"
            age_distribution[age_group] = age_distribution.get(age_group, 0) + 1
        
        # Get anomaly statistics for analyzed drawings
        anomaly_count = db.query(AnomalyAnalysis).filter(AnomalyAnalysis.is_anomaly == True).count()
        normal_count = db.query(AnomalyAnalysis).filter(AnomalyAnalysis.is_anomaly == False).count()
        
        print("=== Drawing Analysis Status ===")
        print(f"Total drawings in database: {total_drawings}")
        print(f"Analyzed drawings: {analyzed_drawings}")
        print(f"Unanalyzed drawings: {len(unanalyzed_drawings)}")
        print(f"Analysis completion: {(analyzed_drawings/total_drawings)*100:.1f}%")
        
        if analyzed_drawings > 0:
            print(f"\n=== Analysis Results ===")
            print(f"Anomalies detected: {anomaly_count}")
            print(f"Normal drawings: {normal_count}")
            print(f"Anomaly rate: {(anomaly_count/(anomaly_count+normal_count))*100:.1f}%")
        
        if unanalyzed_drawings:
            print(f"\n=== Unanalyzed Drawings by Age Group ===")
            for age_group, count in sorted(age_distribution.items()):
                print(f"Age {age_group}: {count} drawings")
            
            print(f"\n=== Sample Unanalyzed Drawings ===")
            for i, drawing in enumerate(unanalyzed_drawings[:10]):
                print(f"{i+1}. ID: {drawing.id}, File: {drawing.filename}, Age: {drawing.age_years}, Subject: {drawing.subject}")
            
            if len(unanalyzed_drawings) > 10:
                print(f"... and {len(unanalyzed_drawings) - 10} more")
        
        print(f"\n=== Next Steps ===")
        if len(unanalyzed_drawings) > 0:
            print("To analyze all unanalyzed drawings:")
            print("  python analyze_all_drawings.py")
            print("  OR")
            print("  python analyze_drawings_direct.py")
        else:
            print("All drawings have been analyzed!")
            print("To re-analyze all drawings:")
            print("  python analyze_all_drawings.py --force")
            print("  OR")
            print("  python analyze_drawings_direct.py --force")
        
    finally:
        db.close()


if __name__ == "__main__":
    check_analysis_status()