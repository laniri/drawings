#!/usr/bin/env python3
"""
Git hooks installer for documentation automation.

This script installs the documentation-related Git hooks:
- pre-commit: Validates documentation changes
- post-commit: Automatically regenerates documentation
- pre-push: Ensures documentation completeness
"""

import os
import sys
import shutil
import stat
from pathlib import Path
from typing import List, Dict, Any


class GitHooksInstaller:
    """Installer for documentation Git hooks."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.hooks_source_dir = Path(__file__).parent
        self.git_hooks_dir = self.project_root / ".git" / "hooks"
        
        # Hook configurations
        self.hooks = {
            'pre-commit': {
                'source': 'pre-commit-docs',
                'description': 'Validates documentation changes before commit'
            },
            'post-commit': {
                'source': 'post-commit-docs', 
                'description': 'Automatically regenerates documentation after commit'
            },
            'pre-push': {
                'source': 'pre-push-docs',
                'description': 'Ensures documentation completeness before push'
            }
        }
    
    def check_git_repository(self) -> bool:
        """Check if we're in a Git repository."""
        if not self.git_hooks_dir.parent.exists():
            print("❌ Not in a Git repository (.git directory not found)")
            print("💡 Initialize Git repository first: git init")
            return False
        
        # Create hooks directory if it doesn't exist
        self.git_hooks_dir.mkdir(exist_ok=True)
        return True
    
    def backup_existing_hooks(self) -> Dict[str, Path]:
        """Backup existing hooks that would be overwritten."""
        backups = {}
        
        for hook_name in self.hooks.keys():
            hook_path = self.git_hooks_dir / hook_name
            
            if hook_path.exists():
                backup_path = self.git_hooks_dir / f"{hook_name}.backup"
                
                # If backup already exists, add timestamp
                if backup_path.exists():
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = self.git_hooks_dir / f"{hook_name}.backup.{timestamp}"
                
                shutil.copy2(hook_path, backup_path)
                backups[hook_name] = backup_path
                print(f"📋 Backed up existing {hook_name} hook to {backup_path.name}")
        
        return backups
    
    def install_hook(self, hook_name: str, hook_config: Dict[str, str]) -> bool:
        """Install a single Git hook."""
        source_path = self.hooks_source_dir / hook_config['source']
        target_path = self.git_hooks_dir / hook_name
        
        if not source_path.exists():
            print(f"❌ Source hook not found: {source_path}")
            return False
        
        try:
            # Copy the hook file
            shutil.copy2(source_path, target_path)
            
            # Make it executable
            current_permissions = target_path.stat().st_mode
            target_path.chmod(current_permissions | stat.S_IEXEC)
            
            print(f"✅ Installed {hook_name} hook")
            print(f"   {hook_config['description']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to install {hook_name} hook: {e}")
            return False
    
    def create_hook_wrapper(self, hook_name: str) -> bool:
        """Create a wrapper script that can handle multiple hooks."""
        target_path = self.git_hooks_dir / hook_name
        
        # Check if there's already a hook that's not ours
        if target_path.exists():
            content = target_path.read_text()
            if 'documentation automation' not in content.lower():
                # There's an existing hook - create a wrapper
                return self._create_multi_hook_wrapper(hook_name)
        
        return True
    
    def _create_multi_hook_wrapper(self, hook_name: str) -> bool:
        """Create a wrapper that can run multiple hooks."""
        target_path = self.git_hooks_dir / hook_name
        docs_hook_path = self.git_hooks_dir / f"{hook_name}-docs"
        original_hook_path = self.git_hooks_dir / f"{hook_name}-original"
        
        try:
            # Move existing hook to -original
            if target_path.exists():
                shutil.move(target_path, original_hook_path)
            
            # Move our docs hook to -docs
            hook_config = self.hooks[hook_name]
            source_path = self.hooks_source_dir / hook_config['source']
            shutil.copy2(source_path, docs_hook_path)
            docs_hook_path.chmod(docs_hook_path.stat().st_mode | stat.S_IEXEC)
            
            # Create wrapper script
            wrapper_content = f'''#!/bin/bash
# Multi-hook wrapper for {hook_name}
# This script runs multiple {hook_name} hooks in sequence

set -e

echo "🔗 Running {hook_name} hooks..."

# Run original hook if it exists
if [ -x "{original_hook_path}" ]; then
    echo "  Running original {hook_name} hook..."
    "{original_hook_path}" "$@"
fi

# Run documentation hook
if [ -x "{docs_hook_path}" ]; then
    echo "  Running documentation {hook_name} hook..."
    "{docs_hook_path}" "$@"
fi

echo "✅ All {hook_name} hooks completed"
'''
            
            target_path.write_text(wrapper_content)
            target_path.chmod(target_path.stat().st_mode | stat.S_IEXEC)
            
            print(f"✅ Created multi-hook wrapper for {hook_name}")
            print(f"   Original hook: {original_hook_path.name}")
            print(f"   Documentation hook: {docs_hook_path.name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create wrapper for {hook_name}: {e}")
            return False
    
    def install_all_hooks(self, force: bool = False) -> bool:
        """Install all documentation hooks."""
        print("🚀 Installing documentation Git hooks...")
        
        if not self.check_git_repository():
            return False
        
        # Backup existing hooks
        if not force:
            backups = self.backup_existing_hooks()
            if backups:
                print(f"📋 Backed up {len(backups)} existing hooks")
        
        # Install each hook
        success_count = 0
        
        for hook_name, hook_config in self.hooks.items():
            print(f"\n📝 Installing {hook_name} hook...")
            
            if self.install_hook(hook_name, hook_config):
                success_count += 1
            else:
                print(f"❌ Failed to install {hook_name} hook")
        
        # Summary
        total_hooks = len(self.hooks)
        if success_count == total_hooks:
            print(f"\n✅ Successfully installed all {total_hooks} documentation hooks!")
            self._print_usage_info()
            return True
        else:
            print(f"\n⚠️  Installed {success_count}/{total_hooks} hooks")
            return False
    
    def uninstall_hooks(self) -> bool:
        """Uninstall documentation hooks."""
        print("🗑️  Uninstalling documentation Git hooks...")
        
        if not self.check_git_repository():
            return False
        
        removed_count = 0
        
        for hook_name in self.hooks.keys():
            hook_path = self.git_hooks_dir / hook_name
            
            if hook_path.exists():
                try:
                    # Check if it's our hook
                    content = hook_path.read_text()
                    if 'documentation' in content.lower():
                        hook_path.unlink()
                        print(f"✅ Removed {hook_name} hook")
                        removed_count += 1
                        
                        # Restore backup if it exists
                        backup_path = self.git_hooks_dir / f"{hook_name}.backup"
                        if backup_path.exists():
                            shutil.move(backup_path, hook_path)
                            print(f"📋 Restored original {hook_name} hook from backup")
                    else:
                        print(f"⚠️  {hook_name} hook exists but doesn't appear to be ours - skipping")
                        
                except Exception as e:
                    print(f"❌ Failed to remove {hook_name} hook: {e}")
            else:
                print(f"📋 {hook_name} hook not found")
        
        print(f"\n✅ Uninstalled {removed_count} documentation hooks")
        return True
    
    def list_hooks(self) -> bool:
        """List installed hooks and their status."""
        print("📋 Documentation Git hooks status:")
        
        if not self.check_git_repository():
            return False
        
        for hook_name, hook_config in self.hooks.items():
            hook_path = self.git_hooks_dir / hook_name
            
            if hook_path.exists():
                try:
                    content = hook_path.read_text()
                    if 'documentation' in content.lower():
                        print(f"  ✅ {hook_name}: Installed")
                        print(f"     {hook_config['description']}")
                    else:
                        print(f"  ⚠️  {hook_name}: Different hook installed")
                except Exception:
                    print(f"  ❓ {hook_name}: Unknown status")
            else:
                print(f"  ❌ {hook_name}: Not installed")
        
        return True
    
    def _print_usage_info(self):
        """Print usage information after installation."""
        print("""
📖 Documentation Git Hooks Usage:

🔍 Pre-commit Hook:
   - Automatically validates documentation changes before commit
   - Checks markdown syntax, links, and documentation standards
   - Ensures code changes have corresponding documentation

🔄 Post-commit Hook:
   - Automatically regenerates documentation after commits
   - Detects which documentation types need updates
   - Can auto-commit generated documentation (set AUTO_COMMIT_DOCS=true)

🚀 Pre-push Hook:
   - Validates documentation completeness before push
   - Checks for broken links and missing documentation
   - Ensures critical documentation exists for new features

💡 Configuration:
   - Set AUTO_COMMIT_DOCS=true to auto-commit generated docs
   - Use 'git commit --no-verify' to skip pre-commit validation
   - Use 'git push --no-verify' to skip pre-push validation

🔧 Management:
   - Install: python scripts/git_hooks/install_hooks.py --install
   - Uninstall: python scripts/git_hooks/install_hooks.py --uninstall
   - Status: python scripts/git_hooks/install_hooks.py --list
""")


def main():
    """Main entry point for hook installer."""
    installer = GitHooksInstaller()
    
    if len(sys.argv) < 2:
        print("📖 Git Hooks Installer for Documentation Automation")
        print("\nUsage:")
        print("  python scripts/git_hooks/install_hooks.py --install    # Install hooks")
        print("  python scripts/git_hooks/install_hooks.py --uninstall  # Uninstall hooks")
        print("  python scripts/git_hooks/install_hooks.py --list       # List hook status")
        print("  python scripts/git_hooks/install_hooks.py --force      # Force install (overwrite)")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    try:
        if command == '--install':
            success = installer.install_all_hooks()
            sys.exit(0 if success else 1)
        elif command == '--force':
            success = installer.install_all_hooks(force=True)
            sys.exit(0 if success else 1)
        elif command == '--uninstall':
            success = installer.uninstall_hooks()
            sys.exit(0 if success else 1)
        elif command == '--list':
            success = installer.list_hooks()
            sys.exit(0 if success else 1)
        else:
            print(f"❌ Unknown command: {command}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Installation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()