#!/usr/bin/env python3
"""
Database Migration CLI Tool

This script provides command-line access to the database migration and backup
functionality for the AWS production deployment system.
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.database_migration_service import database_migration_service


async def create_backup(args):
    """Create a database backup"""
    try:
        print("Creating database backup...")
        
        backup_info = await database_migration_service.create_automated_backup(
            upload_to_s3=args.s3
        )
        
        print(f"✅ Backup created successfully!")
        print(f"   Backup name: {backup_info['backup_name']}")
        print(f"   Backup path: {backup_info['backup_path']}")
        print(f"   Environment: {backup_info['environment']}")
        
        if backup_info.get('s3_uploaded'):
            print(f"   S3 URL: {backup_info['s3_url']}")
        
        if 'migration_info' in backup_info:
            migration_info = backup_info['migration_info']
            print(f"   Current revision: {migration_info.get('current_revision', 'Unknown')}")
            print(f"   Up to date: {migration_info.get('is_up_to_date', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Backup failed: {str(e)}")
        sys.exit(1)


async def run_migration(args):
    """Run database migration"""
    try:
        print(f"Running migration to {args.revision}...")
        
        migration_result = await database_migration_service.run_migrations(
            target_revision=args.revision
        )
        
        print(f"✅ Migration completed successfully!")
        print(f"   Target revision: {migration_result['target_revision']}")
        print(f"   Current revision: {migration_result['post_migration']['current_revision']}")
        print(f"   Backup created: {migration_result['backup_info']['backup_name']}")
        
        consistency = migration_result['consistency_check']
        if consistency['status'] == 'passed':
            print(f"   ✅ Consistency check: PASSED")
        else:
            print(f"   ⚠️  Consistency check: {consistency['status']}")
            if consistency.get('errors'):
                for error in consistency['errors']:
                    print(f"      - {error}")
        
    except Exception as e:
        print(f"❌ Migration failed: {str(e)}")
        sys.exit(1)


async def check_migration_info(args):
    """Check current migration information"""
    try:
        print("Checking migration information...")
        
        migration_info = await database_migration_service._get_migration_info()
        
        print(f"📊 Migration Information:")
        print(f"   Current revision: {migration_info.get('current_revision', 'Unknown')}")
        print(f"   Head revision: {migration_info.get('head_revision', 'Unknown')}")
        print(f"   Up to date: {migration_info.get('is_up_to_date', 'Unknown')}")
        
        if migration_info.get('error'):
            print(f"   ⚠️  Error: {migration_info['error']}")
        
    except Exception as e:
        print(f"❌ Failed to get migration info: {str(e)}")
        sys.exit(1)


async def validate_consistency(args):
    """Validate database consistency"""
    try:
        print("Running database consistency check...")
        
        consistency_result = await database_migration_service._validate_migration_consistency()
        
        print(f"🔍 Consistency Check Results:")
        print(f"   Status: {consistency_result['status']}")
        print(f"   Foreign keys enabled: {consistency_result.get('foreign_keys_enabled', 'Unknown')}")
        print(f"   Integrity check passed: {consistency_result.get('integrity_check_passed', 'Unknown')}")
        print(f"   Table count: {consistency_result.get('table_count', 'Unknown')}")
        
        if consistency_result.get('errors'):
            print(f"   ⚠️  Errors found:")
            for error in consistency_result['errors']:
                print(f"      - {error}")
        else:
            print(f"   ✅ No errors found")
        
    except Exception as e:
        print(f"❌ Consistency check failed: {str(e)}")
        sys.exit(1)


async def list_backups(args):
    """List available backups"""
    try:
        print("Listing available backups...")
        
        backup_list = await database_migration_service.backup_service.get_backup_list()
        
        if not backup_list:
            print("📁 No backups found")
            return
        
        print(f"📁 Available Backups ({len(backup_list)} found):")
        for backup in backup_list:
            print(f"   📄 {backup['name']}")
            print(f"      Size: {backup['size_mb']} MB")
            print(f"      Created: {backup['created']}")
            print(f"      Type: {backup['type']}")
            print()
        
    except Exception as e:
        print(f"❌ Failed to list backups: {str(e)}")
        sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Database Migration CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s backup                    # Create local backup
  %(prog)s backup --s3               # Create backup and upload to S3
  %(prog)s migrate                   # Run migrations to head
  %(prog)s migrate --revision abc123 # Run migrations to specific revision
  %(prog)s info                      # Show migration information
  %(prog)s check                     # Run consistency check
  %(prog)s list                      # List available backups
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Create database backup')
    backup_parser.add_argument('--s3', action='store_true', 
                              help='Upload backup to S3 (production only)')
    
    # Migration command
    migrate_parser = subparsers.add_parser('migrate', help='Run database migration')
    migrate_parser.add_argument('--revision', default='head',
                               help='Target revision (default: head)')
    
    # Info command
    subparsers.add_parser('info', help='Show migration information')
    
    # Consistency check command
    subparsers.add_parser('check', help='Run database consistency check')
    
    # List backups command
    subparsers.add_parser('list', help='List available backups')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Print environment info
    env_config = database_migration_service.env_config
    print(f"🌍 Environment: {env_config.environment.value}")
    print(f"🗄️  Database: {env_config.database_url}")
    if env_config.s3_bucket_name:
        print(f"☁️  S3 Bucket: {env_config.s3_bucket_name}")
    print()
    
    # Run the appropriate command
    if args.command == 'backup':
        asyncio.run(create_backup(args))
    elif args.command == 'migrate':
        asyncio.run(run_migration(args))
    elif args.command == 'info':
        asyncio.run(check_migration_info(args))
    elif args.command == 'check':
        asyncio.run(validate_consistency(args))
    elif args.command == 'list':
        asyncio.run(list_backups(args))


if __name__ == '__main__':
    main()