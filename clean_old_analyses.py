#!/usr/bin/env python3
"""
Clean old analyses from the database.

This script removes all existing anomaly analyses and interpretability results
so that drawings can be re-analyzed with the corrected explanation text logic.
"""

import sys
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.database import AnomalyAnalysis, InterpretabilityResult

def clean_old_analyses():
    """Remove all existing analyses and interpretability results."""
    
    print("🧹 Cleaning old analyses from database...")
    
    try:
        db: Session = next(get_db())
        
        # Count existing records
        analysis_count = db.query(AnomalyAnalysis).count()
        interpretability_count = db.query(InterpretabilityResult).count()
        
        print(f"📊 Found {analysis_count} analyses and {interpretability_count} interpretability results")
        
        if analysis_count == 0 and interpretability_count == 0:
            print("✅ Database is already clean - no analyses to remove")
            return
        
        # Confirm deletion
        response = input(f"\n⚠️  This will permanently delete {analysis_count} analyses and {interpretability_count} interpretability results.\nAre you sure? (yes/no): ")
        
        if response.lower() != 'yes':
            print("❌ Operation cancelled")
            return
        
        print("\n🗑️  Deleting records...")
        
        # Delete interpretability results first (they reference analyses)
        if interpretability_count > 0:
            deleted_interpretability = db.query(InterpretabilityResult).delete()
            print(f"   Deleted {deleted_interpretability} interpretability results")
        
        # Delete analyses
        if analysis_count > 0:
            deleted_analyses = db.query(AnomalyAnalysis).delete()
            print(f"   Deleted {deleted_analyses} analyses")
        
        # Commit the changes
        db.commit()
        
        print("\n✅ Successfully cleaned old analyses from database")
        print("💡 You can now re-analyze drawings to get corrected explanation text")
        print("\n📝 To re-analyze all drawings, run:")
        print("   python analyze_all_drawings.py --force")
        print("\n📝 To test a specific drawing, run:")
        print("   python test_analysis_fix.py --drawing-id 1")
        
    except Exception as e:
        print(f"❌ Error cleaning database: {str(e)}")
        if 'db' in locals():
            db.rollback()
        sys.exit(1)
    
    finally:
        if 'db' in locals():
            db.close()

def main():
    """Main function."""
    print("🔧 Database Cleanup Tool")
    print("=" * 50)
    clean_old_analyses()

if __name__ == "__main__":
    main()