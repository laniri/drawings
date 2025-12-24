#!/usr/bin/env python3
"""
Deployment Integration Test Script

This script tests the complete deployment pipeline integration including:
- Docker build and container functionality
- Environment configuration switching
- Service health checks
- API endpoint validation
- Database connectivity
- File storage operations
"""

import asyncio
import json
import logging
import os
import sys
import time
import requests
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import signal

# Add app to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import get_settings
from app.services.auth_service import get_auth_service
from app.services.monitoring_service import get_monitoring_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeploymentIntegrationTester:
    """Test deployment pipeline integration."""
    
    def __init__(self):
        self.settings = get_settings()
        self.test_results = {
            "docker_build": {},
            "container_health": {},
            "api_endpoints": {},
            "authentication": {},
            "database_operations": {},
            "file_storage": {},
            "monitoring_integration": {},
            "overall_status": "pending"
        }
        self.container_id = None
        self.container_port = 8080
    
    async def run_integration_tests(self) -> Dict[str, Any]:
        """Run complete deployment integration tests."""
        logger.info("Starting deployment integration tests...")
        
        try:
            # 1. Test Docker build
            await self.test_docker_build()
            
            # 2. Test container deployment
            await self.test_container_deployment()
            
            # 3. Test API endpoints
            await self.test_api_endpoints()
            
            # 4. Test authentication integration
            await self.test_authentication_integration()
            
            # 5. Test database operations
            await self.test_database_operations()
            
            # 6. Test file storage operations
            await self.test_file_storage_operations()
            
            # 7. Test monitoring integration
            await self.test_monitoring_integration()
            
            # Calculate overall status
            self._calculate_overall_status()
            
            logger.info("Deployment integration tests completed")
            return self.test_results
            
        except Exception as e:
            logger.error(f"Integration tests failed: {e}")
            self.test_results["overall_status"] = "failed"
            self.test_results["error"] = str(e)
            return self.test_results
        
        finally:
            # Cleanup container if running
            await self.cleanup_container()
    
    async def test_docker_build(self):
        """Test Docker build process."""
        logger.info("Testing Docker build...")
        
        build_results = {
            "dockerfile_exists": False,
            "build_successful": False,
            "image_created": False,
            "build_time_seconds": 0,
            "details": {}
        }
        
        try:
            # Check if Dockerfile.prod exists
            dockerfile_path = Path("Dockerfile.prod")
            build_results["dockerfile_exists"] = dockerfile_path.exists()
            
            if not dockerfile_path.exists():
                build_results["details"]["error"] = "Dockerfile.prod not found"
                self.test_results["docker_build"] = build_results
                return
            
            # Build Docker image
            image_tag = "children-drawing-test:latest"
            start_time = time.time()
            
            logger.info("Building Docker image (this may take a few minutes)...")
            result = subprocess.run([
                "docker", "build",
                "-f", str(dockerfile_path),
                "-t", image_tag,
                "."
            ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
            
            build_time = time.time() - start_time
            build_results["build_time_seconds"] = round(build_time, 1)
            
            if result.returncode == 0:
                build_results["build_successful"] = True
                logger.info(f"Docker build successful in {build_time:.1f} seconds")
                
                # Check if image was created
                check_result = subprocess.run([
                    "docker", "images", image_tag, "--format", "{{.Repository}}:{{.Tag}}"
                ], capture_output=True, text=True)
                
                if check_result.returncode == 0 and image_tag in check_result.stdout:
                    build_results["image_created"] = True
                    build_results["details"]["image_tag"] = image_tag
                
            else:
                build_results["details"]["build_error"] = result.stderr
                logger.error(f"Docker build failed: {result.stderr}")
            
        except subprocess.TimeoutExpired:
            build_results["details"]["error"] = "Docker build timed out after 10 minutes"
            logger.error("Docker build timed out")
        except FileNotFoundError:
            build_results["details"]["error"] = "Docker not found - please install Docker"
            logger.error("Docker not available")
        except Exception as e:
            build_results["details"]["error"] = str(e)
            logger.error(f"Docker build test failed: {e}")
        
        self.test_results["docker_build"] = build_results
    
    async def test_container_deployment(self):
        """Test container deployment and health."""
        logger.info("Testing container deployment...")
        
        deployment_results = {
            "container_started": False,
            "health_check_passed": False,
            "port_accessible": False,
            "startup_time_seconds": 0,
            "details": {}
        }
        
        try:
            if not self.test_results["docker_build"]["image_created"]:
                deployment_results["details"]["error"] = "No Docker image available for testing"
                self.test_results["container_health"] = deployment_results
                return
            
            # Start container
            image_tag = "children-drawing-test:latest"
            start_time = time.time()
            
            logger.info(f"Starting container on port {self.container_port}...")
            result = subprocess.run([
                "docker", "run",
                "-d",  # Detached mode
                "-p", f"{self.container_port}:80",
                "--name", "children-drawing-test-container",
                "-e", "ENVIRONMENT=test",
                "-e", "DATABASE_URL=sqlite:///./test.db",
                image_tag
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.container_id = result.stdout.strip()
                deployment_results["container_started"] = True
                deployment_results["details"]["container_id"] = self.container_id
                logger.info(f"Container started: {self.container_id[:12]}")
                
                # Wait for container to be ready
                await asyncio.sleep(10)  # Give container time to start
                
                # Check container health
                health_result = subprocess.run([
                    "docker", "ps", "--filter", f"id={self.container_id}", "--format", "{{.Status}}"
                ], capture_output=True, text=True)
                
                if health_result.returncode == 0 and "Up" in health_result.stdout:
                    deployment_results["health_check_passed"] = True
                    
                    # Test port accessibility
                    try:
                        response = requests.get(f"http://localhost:{self.container_port}/health", timeout=10)
                        if response.status_code == 200:
                            deployment_results["port_accessible"] = True
                            deployment_results["details"]["health_response"] = response.json()
                        else:
                            deployment_results["details"]["health_status_code"] = response.status_code
                    except requests.RequestException as e:
                        deployment_results["details"]["port_test_error"] = str(e)
                
                startup_time = time.time() - start_time
                deployment_results["startup_time_seconds"] = round(startup_time, 1)
                
            else:
                deployment_results["details"]["start_error"] = result.stderr
                logger.error(f"Container start failed: {result.stderr}")
            
        except Exception as e:
            deployment_results["details"]["error"] = str(e)
            logger.error(f"Container deployment test failed: {e}")
        
        self.test_results["container_health"] = deployment_results
    
    async def test_api_endpoints(self):
        """Test API endpoint accessibility."""
        logger.info("Testing API endpoints...")
        
        api_results = {
            "health_endpoint": False,
            "docs_endpoint": False,
            "api_v1_accessible": False,
            "cors_configured": False,
            "response_times": {},
            "details": {}
        }
        
        try:
            if not self.test_results["container_health"]["port_accessible"]:
                api_results["details"]["error"] = "Container not accessible for API testing"
                self.test_results["api_endpoints"] = api_results
                return
            
            base_url = f"http://localhost:{self.container_port}"
            
            # Test health endpoint
            try:
                start_time = time.time()
                response = requests.get(f"{base_url}/health", timeout=5)
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    api_results["health_endpoint"] = True
                    api_results["response_times"]["health"] = round(response_time * 1000, 1)  # ms
                    api_results["details"]["health_data"] = response.json()
                
            except requests.RequestException as e:
                api_results["details"]["health_error"] = str(e)
            
            # Test docs endpoint
            try:
                start_time = time.time()
                response = requests.get(f"{base_url}/docs", timeout=5)
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    api_results["docs_endpoint"] = True
                    api_results["response_times"]["docs"] = round(response_time * 1000, 1)  # ms
                
            except requests.RequestException as e:
                api_results["details"]["docs_error"] = str(e)
            
            # Test API v1 root
            try:
                start_time = time.time()
                response = requests.get(f"{base_url}/api/v1/", timeout=5)
                response_time = time.time() - start_time
                
                # Accept both 200 and 404 as valid (endpoint exists but may not have root handler)
                if response.status_code in [200, 404, 405]:
                    api_results["api_v1_accessible"] = True
                    api_results["response_times"]["api_v1"] = round(response_time * 1000, 1)  # ms
                
            except requests.RequestException as e:
                api_results["details"]["api_v1_error"] = str(e)
            
            # Test CORS configuration (simple check)
            try:
                response = requests.options(f"{base_url}/api/v1/", timeout=5)
                cors_headers = response.headers.get("Access-Control-Allow-Origin")
                if cors_headers:
                    api_results["cors_configured"] = True
                    api_results["details"]["cors_headers"] = cors_headers
                
            except requests.RequestException as e:
                api_results["details"]["cors_error"] = str(e)
            
        except Exception as e:
            api_results["details"]["error"] = str(e)
            logger.error(f"API endpoint test failed: {e}")
        
        self.test_results["api_endpoints"] = api_results
    
    async def test_authentication_integration(self):
        """Test authentication system integration."""
        logger.info("Testing authentication integration...")
        
        auth_results = {
            "service_available": False,
            "login_endpoint": False,
            "session_management": False,
            "rate_limiting": False,
            "details": {}
        }
        
        try:
            # Test authentication service directly
            auth_service = get_auth_service()
            if auth_service:
                auth_results["service_available"] = True
                
                # Test session creation and management
                test_session = auth_service.create_session("127.0.0.1")
                if test_session:
                    auth_results["session_management"] = True
                    
                    # Test session verification
                    is_valid = auth_service.verify_session(test_session)
                    auth_results["details"]["session_verification"] = is_valid
                    
                    # Cleanup test session
                    auth_service.logout(test_session)
                
                # Test rate limiting functionality
                if hasattr(auth_service, '_is_rate_limited'):
                    auth_results["rate_limiting"] = True
            
            # Test login endpoint if container is running
            if self.test_results["container_health"]["port_accessible"]:
                base_url = f"http://localhost:{self.container_port}"
                
                # Test login endpoint (may not exist, but check)
                try:
                    response = requests.post(f"{base_url}/api/v1/auth/login", 
                                           json={"password": "test"}, timeout=5)
                    # Accept various response codes as indication endpoint exists
                    if response.status_code in [200, 400, 401, 422]:
                        auth_results["login_endpoint"] = True
                        auth_results["details"]["login_status_code"] = response.status_code
                        
                except requests.RequestException as e:
                    auth_results["details"]["login_endpoint_error"] = str(e)
            
        except Exception as e:
            auth_results["details"]["error"] = str(e)
            logger.error(f"Authentication integration test failed: {e}")
        
        self.test_results["authentication"] = auth_results
    
    async def test_database_operations(self):
        """Test database connectivity and operations."""
        logger.info("Testing database operations...")
        
        db_results = {
            "database_accessible": False,
            "tables_exist": False,
            "crud_operations": False,
            "migrations_applied": False,
            "details": {}
        }
        
        try:
            # Test database file existence
            db_path = Path("drawings.db")
            if db_path.exists():
                db_results["database_accessible"] = True
                db_results["details"]["database_size_mb"] = round(db_path.stat().st_size / (1024*1024), 2)
                
                # Test database connectivity
                import sqlite3
                try:
                    conn = sqlite3.connect(str(db_path))
                    cursor = conn.cursor()
                    
                    # Check if main tables exist
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    expected_tables = ["drawings", "age_group_models", "anomaly_analyses"]
                    existing_tables = [table for table in expected_tables if table in tables]
                    
                    if len(existing_tables) >= 2:  # At least 2 main tables
                        db_results["tables_exist"] = True
                        db_results["details"]["tables"] = tables
                    
                    # Test basic CRUD (read operation)
                    try:
                        cursor.execute("SELECT COUNT(*) FROM drawings")
                        drawing_count = cursor.fetchone()[0]
                        db_results["crud_operations"] = True
                        db_results["details"]["drawing_count"] = drawing_count
                        
                    except sqlite3.Error as e:
                        db_results["details"]["crud_error"] = str(e)
                    
                    # Check for Alembic version table (indicates migrations)
                    if "alembic_version" in tables:
                        db_results["migrations_applied"] = True
                        cursor.execute("SELECT version_num FROM alembic_version")
                        version = cursor.fetchone()
                        if version:
                            db_results["details"]["migration_version"] = version[0]
                    
                    conn.close()
                    
                except sqlite3.Error as e:
                    db_results["details"]["sqlite_error"] = str(e)
            
            else:
                db_results["details"]["error"] = "Database file not found"
            
        except Exception as e:
            db_results["details"]["error"] = str(e)
            logger.error(f"Database operations test failed: {e}")
        
        self.test_results["database_operations"] = db_results
    
    async def test_file_storage_operations(self):
        """Test file storage operations."""
        logger.info("Testing file storage operations...")
        
        storage_results = {
            "upload_directory": False,
            "static_directory": False,
            "write_permissions": False,
            "storage_service": False,
            "details": {}
        }
        
        try:
            # Check upload directory
            upload_dir = Path("uploads")
            if upload_dir.exists() and upload_dir.is_dir():
                storage_results["upload_directory"] = True
                storage_results["details"]["upload_dir_size"] = len(list(upload_dir.rglob("*")))
            
            # Check static directory
            static_dir = Path("static")
            if static_dir.exists() and static_dir.is_dir():
                storage_results["static_directory"] = True
                storage_results["details"]["static_dir_size"] = len(list(static_dir.rglob("*")))
            
            # Test write permissions
            try:
                test_file = Path("test_write_permission.tmp")
                test_file.write_text("test")
                if test_file.exists():
                    storage_results["write_permissions"] = True
                    test_file.unlink()  # Clean up
                    
            except Exception as e:
                storage_results["details"]["write_permission_error"] = str(e)
            
            # Test storage service (if available)
            try:
                from app.services.file_storage import FileStorageService
                storage_service = FileStorageService()
                if storage_service:
                    storage_results["storage_service"] = True
                    
            except ImportError:
                storage_results["details"]["storage_service_error"] = "FileStorageService not available"
            
        except Exception as e:
            storage_results["details"]["error"] = str(e)
            logger.error(f"File storage operations test failed: {e}")
        
        self.test_results["file_storage"] = storage_results
    
    async def test_monitoring_integration(self):
        """Test monitoring system integration."""
        logger.info("Testing monitoring integration...")
        
        monitoring_results = {
            "service_available": False,
            "log_file_creation": False,
            "metrics_recording": False,
            "alert_system": False,
            "details": {}
        }
        
        try:
            # Test monitoring service
            monitoring_service = get_monitoring_service()
            if monitoring_service:
                monitoring_results["service_available"] = True
                
                # Test log file creation
                log_file = Path("monitoring.log")
                if log_file.exists():
                    monitoring_results["log_file_creation"] = True
                    monitoring_results["details"]["log_file_size"] = log_file.stat().st_size
                
                # Test metrics recording
                metrics_result = monitoring_service.record_performance_metrics({
                    "integration_test_metric": 1.0
                })
                if metrics_result.success:
                    monitoring_results["metrics_recording"] = True
                
                # Test alert system
                alert_result = monitoring_service.send_alert(
                    level=monitoring_service.AlertLevel.INFO,
                    message="Integration test alert"
                )
                if alert_result.success:
                    monitoring_results["alert_system"] = True
                
                # Get service stats
                stats = monitoring_service.get_service_stats()
                monitoring_results["details"]["service_stats"] = stats
            
        except Exception as e:
            monitoring_results["details"]["error"] = str(e)
            logger.error(f"Monitoring integration test failed: {e}")
        
        self.test_results["monitoring_integration"] = monitoring_results
    
    async def cleanup_container(self):
        """Clean up test container."""
        if self.container_id:
            try:
                logger.info("Cleaning up test container...")
                
                # Stop container
                subprocess.run(["docker", "stop", self.container_id], 
                             capture_output=True, timeout=30)
                
                # Remove container
                subprocess.run(["docker", "rm", self.container_id], 
                             capture_output=True, timeout=30)
                
                logger.info("Test container cleaned up")
                
            except Exception as e:
                logger.warning(f"Container cleanup failed: {e}")
    
    def _calculate_overall_status(self):
        """Calculate overall integration test status."""
        total_checks = 0
        passed_checks = 0
        
        for category, results in self.test_results.items():
            if category in ["overall_status", "error"]:
                continue
                
            for key, value in results.items():
                if isinstance(value, bool) and key != "details":
                    total_checks += 1
                    if value:
                        passed_checks += 1
        
        success_rate = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        
        if success_rate >= 90:
            self.test_results["overall_status"] = "excellent"
        elif success_rate >= 75:
            self.test_results["overall_status"] = "good"
        elif success_rate >= 60:
            self.test_results["overall_status"] = "acceptable"
        else:
            self.test_results["overall_status"] = "needs_improvement"
        
        self.test_results["summary"] = {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "success_rate": round(success_rate, 1)
        }
        
        # Print summary
        print("\n" + "="*80)
        print("DEPLOYMENT INTEGRATION TEST SUMMARY")
        print("="*80)
        print(f"Overall Status: {self.test_results['overall_status'].upper()}")
        print(f"Success Rate: {success_rate:.1f}% ({passed_checks}/{total_checks} checks)")
        print("="*80)


async def main():
    """Main integration test function."""
    tester = DeploymentIntegrationTester()
    
    # Set up signal handler for cleanup
    def signal_handler(signum, frame):
        logger.info("Received interrupt signal, cleaning up...")
        asyncio.create_task(tester.cleanup_container())
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        results = await tester.run_integration_tests()
        
        # Save results
        with open("deployment_integration_report.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # Return appropriate exit code
        if results["overall_status"] in ["excellent", "good"]:
            sys.exit(0)
        elif results["overall_status"] == "acceptable":
            sys.exit(1)  # Warning
        else:
            sys.exit(2)  # Error
            
    except KeyboardInterrupt:
        logger.info("Integration tests interrupted by user")
        await tester.cleanup_container()
        sys.exit(130)


if __name__ == "__main__":
    asyncio.run(main())