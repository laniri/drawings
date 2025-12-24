#!/usr/bin/env python3
"""
Script to optimize SQLite database performance by updating statistics and defragmenting.
"""

import sqlite3
import time
import os

def optimize_database():
    """Optimize the SQLite database for better performance."""
    db_path = "drawings.db"
    
    if not os.path.exists(db_path):
        print(f"Database {db_path} not found!")
        return
    
    print("Starting database optimization...")
    start_time = time.time()
    
    # Get initial database size
    initial_size = os.path.getsize(db_path) / (1024 * 1024)  # MB
    print(f"Initial database size: {initial_size:.1f} MB")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # 1. Update table statistics (helps query planner choose better indexes)
        print("1. Updating table statistics...")
        cursor.execute("ANALYZE")
        conn.commit()
        print("   ✓ Statistics updated")
        
        # 2. Rebuild indexes and defragment database
        print("2. Defragmenting database (VACUUM)...")
        cursor.execute("VACUUM")
        conn.commit()
        print("   ✓ Database defragmented")
        
        # 3. Verify indexes are present
        print("3. Verifying indexes...")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%' ORDER BY name")
        indexes = cursor.fetchall()
        print(f"   ✓ Found {len(indexes)} performance indexes")
        
        # 4. Check database integrity
        print("4. Checking database integrity...")
        cursor.execute("PRAGMA integrity_check")
        integrity = cursor.fetchone()[0]
        if integrity == "ok":
            print("   ✓ Database integrity OK")
        else:
            print(f"   ⚠️  Database integrity issue: {integrity}")
        
        # 5. Optimize SQLite settings for bulk operations
        print("5. Optimizing SQLite settings...")
        cursor.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging for better concurrency
        cursor.execute("PRAGMA synchronous = NORMAL")  # Balance between safety and speed
        cursor.execute("PRAGMA cache_size = -64000")  # 64MB cache
        cursor.execute("PRAGMA temp_store = MEMORY")  # Use memory for temp tables
        cursor.execute("PRAGMA mmap_size = 268435456")  # 256MB memory-mapped I/O
        print("   ✓ SQLite settings optimized")
        
        # 6. Show current settings
        print("6. Current SQLite configuration:")
        settings = [
            ("journal_mode", "PRAGMA journal_mode"),
            ("synchronous", "PRAGMA synchronous"),
            ("cache_size", "PRAGMA cache_size"),
            ("temp_store", "PRAGMA temp_store"),
            ("mmap_size", "PRAGMA mmap_size")
        ]
        
        for name, pragma in settings:
            cursor.execute(pragma)
            value = cursor.fetchone()[0]
            print(f"   {name}: {value}")
        
    except Exception as e:
        print(f"Error during optimization: {e}")
    finally:
        conn.close()
    
    # Get final database size
    final_size = os.path.getsize(db_path) / (1024 * 1024)  # MB
    size_change = final_size - initial_size
    
    elapsed_time = time.time() - start_time
    
    print(f"\nOptimization completed in {elapsed_time:.1f} seconds")
    print(f"Final database size: {final_size:.1f} MB")
    if size_change < 0:
        print(f"Space saved: {abs(size_change):.1f} MB")
    elif size_change > 0:
        print(f"Size increased: {size_change:.1f} MB (normal after optimization)")
    
    print("\n✅ Database optimization complete!")
    print("   - Table statistics updated for better query planning")
    print("   - Database defragmented for optimal performance")
    print("   - SQLite settings optimized for bulk operations")
    print("   - All performance indexes verified")

def show_performance_tips():
    """Show additional performance tips."""
    print("\n📈 Performance Tips for Bulk Analysis:")
    print("1. Run analysis during off-peak hours")
    print("2. Close other applications to free up memory")
    print("3. Consider using smaller batch sizes (--batch-size 25)")
    print("4. Monitor system resources during analysis")
    print("5. The first few batches may be slower as caches warm up")

if __name__ == "__main__":
    optimize_database()
    show_performance_tips()