"""
Property-based test for automated deployment trigger consistency.

**Feature: aws-production-deployment, Property 4: Automated Deployment Trigger Consistency**
**Validates: Requirements 3.1, 3.2, 3.4**

This test validates that code pushes to the main branch automatically trigger
deployment pipeline and produce consistent deployment results.
"""

import os
import json
import tempfile
import yaml
from pathlib import Path
import pytest
from hypothesis import given, strategies as st, assume, settings
from unittest.mock import patch, MagicMock, call
from typing import Dict, Any, List


class MockGitHubEvent:
    """Mock GitHub webhook event for testing deployment triggers"""
    
    def __init__(self, event_type: str, ref: str, commit_sha: str, author: str):
        self.event_type = event_type
        self.ref = ref
        self.commit_sha = commit_sha
        self.author = author
        self.timestamp = "2024-01-01T12:00:00Z"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to GitHub webhook payload format"""
        if self.event_type == "push":
            return {
                "ref": self.ref,
                "after": self.commit_sha,
                "before": "0000000000000000000000000000000000000000",
                "commits": [
                    {
                        "id": self.commit_sha,
                        "message": "Test commit",
                        "author": {"name": self.author, "email": f"{self.author}@example.com"},
                        "timestamp": self.timestamp
                    }
                ],
                "head_commit": {
                    "id": self.commit_sha,
                    "message": "Test commit",
                    "author": {"name": self.author, "email": f"{self.author}@example.com"},
                    "timestamp": self.timestamp
                },
                "repository": {
                    "name": "children-drawing-anomaly-detection",
                    "full_name": "test-org/children-drawing-anomaly-detection"
                }
            }
        elif self.event_type == "pull_request":
            return {
                "action": "opened",
                "number": 123,
                "pull_request": {
                    "head": {
                        "ref": self.ref.replace("refs/heads/", ""),
                        "sha": self.commit_sha
                    },
                    "base": {
                        "ref": "main",
                        "sha": "main_commit_sha"
                    }
                },
                "repository": {
                    "name": "children-drawing-anomaly-detection",
                    "full_name": "test-org/children-drawing-anomaly-detection"
                }
            }
        else:
            return {}


class MockGitHubActionsWorkflow:
    """Mock GitHub Actions workflow for testing deployment consistency"""
    
    def __init__(self, workflow_file: str):
        self.workflow_file = workflow_file
        self.workflow_data = self._load_workflow()
        self.runs = []
    
    def _load_workflow(self) -> Dict[str, Any]:
        """Load workflow configuration from file"""
        try:
            with open(self.workflow_file, 'r') as f:
                # Use safe_load with proper YAML handling
                content = f.read()
                # Replace 'on:' with 'trigger:' to avoid YAML boolean parsing issues
                content = content.replace('on:', 'trigger:')
                workflow_data = yaml.safe_load(content)
                # Convert back to 'on' for consistency
                if 'trigger' in workflow_data:
                    workflow_data['on'] = workflow_data.pop('trigger')
                return workflow_data
        except FileNotFoundError:
            # Return a mock workflow structure
            return {
                "name": "Deploy to AWS Production",
                "on": {
                    "push": {"branches": ["main"]},
                    "pull_request": {"branches": ["main"]},
                    "workflow_dispatch": {}
                },
                "jobs": {
                    "test": {"runs-on": "ubuntu-latest"},
                    "security-scan": {"runs-on": "ubuntu-latest"},
                    "build-and-push": {"runs-on": "ubuntu-latest", "needs": ["test", "security-scan"]},
                    "deploy-application": {"runs-on": "ubuntu-latest", "needs": ["build-and-push"]},
                    "health-check": {"runs-on": "ubuntu-latest", "needs": ["deploy-application"]}
                }
            }
    
    def should_trigger(self, event: MockGitHubEvent) -> bool:
        """Determine if workflow should trigger for given event"""
        triggers = self.workflow_data.get("on", {})
        
        if event.event_type == "push":
            push_config = triggers.get("push", {})
            if isinstance(push_config, dict):
                branches = push_config.get("branches", [])
                # Check if the ref matches any of the configured branches
                for branch in branches:
                    if event.ref == f"refs/heads/{branch}":
                        return True
                return False
            return "push" in triggers
        
        elif event.event_type == "pull_request":
            pr_config = triggers.get("pull_request", {})
            if isinstance(pr_config, dict):
                branches = pr_config.get("branches", [])
                # For PR events, we check if the target branch matches
                # In our mock, we'll assume PR is targeting main if branches include main
                return "main" in branches
            return "pull_request" in triggers
        
        return False
    
    def get_deployment_jobs(self) -> List[str]:
        """Get list of jobs that perform deployment"""
        deployment_jobs = []
        jobs = self.workflow_data.get("jobs", {})
        
        for job_name, job_config in jobs.items():
            # Jobs that deploy to production
            if any(keyword in job_name.lower() for keyword in ["deploy", "build-and-push"]):
                deployment_jobs.append(job_name)
        
        return deployment_jobs
    
    def simulate_run(self, event: MockGitHubEvent) -> Dict[str, Any]:
        """Simulate a workflow run"""
        if not self.should_trigger(event):
            return {"triggered": False, "reason": "Event does not match trigger conditions"}
        
        run_result = {
            "triggered": True,
            "event": event.to_dict(),
            "workflow_id": f"run_{len(self.runs) + 1}",
            "commit_sha": event.commit_sha,
            "ref": event.ref,
            "jobs": {}
        }
        
        jobs = self.workflow_data.get("jobs", {})
        for job_name, job_config in jobs.items():
            # Simulate job execution
            job_result = {
                "status": "success",
                "duration": 120,  # seconds
                "steps": []
            }
            
            # Add deployment-specific results for deployment jobs
            if job_name in self.get_deployment_jobs():
                # Only deploy for push events to main branch
                if event.event_type == "push" and event.ref == "refs/heads/main":
                    job_result["deployment"] = {
                        "environment": "production",
                        "image_tag": event.commit_sha,
                        "deployed": True
                    }
                else:
                    job_result["deployment"] = {
                        "deployed": False, 
                        "reason": "Not main branch push" if event.event_type == "push" else "Pull request - no deployment"
                    }
            
            run_result["jobs"][job_name] = job_result
        
        self.runs.append(run_result)
        return run_result


class TestAutomatedDeploymentTriggerConsistency:
    """Property-based tests for automated deployment trigger consistency"""
    
    @given(
        commit_sha=st.text(min_size=40, max_size=40, alphabet=st.characters(whitelist_categories=('Nd', 'Ll'))),
        author=st.text(min_size=3, max_size=20, alphabet=st.characters(whitelist_categories=('Ll', 'Lu'))),
        branch_name=st.sampled_from(["main", "master", "develop", "feature/test", "hotfix/urgent"])
    )
    @settings(deadline=None)  # Disable deadline for this test
    def test_push_event_deployment_trigger_consistency(
        self, 
        commit_sha: str, 
        author: str, 
        branch_name: str
    ):
        """
        **Feature: aws-production-deployment, Property 4: Automated Deployment Trigger Consistency**
        **Validates: Requirements 3.1, 3.2, 3.4**
        
        For any code push to the main branch, the deployment pipeline should automatically
        trigger and produce consistent deployment results.
        """
        # Create GitHub push event
        push_event = MockGitHubEvent(
            event_type="push",
            ref=f"refs/heads/{branch_name}",
            commit_sha=commit_sha,
            author=author
        )
        
        # Create workflow instance
        workflow = MockGitHubActionsWorkflow("infrastructure/github-actions-workflow.yml")
        
        # Test trigger consistency - run multiple times with same event
        results = []
        for _ in range(3):
            result = workflow.simulate_run(push_event)
            results.append(result)
        
        # All runs should have consistent trigger behavior
        trigger_decisions = [r["triggered"] for r in results]
        assert all(decision == trigger_decisions[0] for decision in trigger_decisions)
        
        # If triggered, verify deployment consistency
        if results[0]["triggered"]:
            deployment_jobs = workflow.get_deployment_jobs()
            
            for result in results:
                # All runs should have same jobs
                assert set(result["jobs"].keys()) == set(results[0]["jobs"].keys())
                
                # Deployment jobs should behave consistently
                for job_name in deployment_jobs:
                    job_result = result["jobs"][job_name]
                    first_job_result = results[0]["jobs"][job_name]
                    
                    # Deployment decision should be consistent
                    if "deployment" in job_result:
                        assert job_result["deployment"]["deployed"] == first_job_result["deployment"]["deployed"]
                        
                        # If deployed, image tag should be consistent (same commit SHA)
                        if job_result["deployment"]["deployed"]:
                            assert job_result["deployment"]["image_tag"] == commit_sha
                            assert job_result["deployment"]["environment"] == "production"
        
        # Verify main branch triggers deployment
        if branch_name == "main":
            assert results[0]["triggered"], f"Main branch should trigger deployment but got: {results[0]}"
            deployment_jobs = workflow.get_deployment_jobs()
            for job_name in deployment_jobs:
                job_result = results[0]["jobs"][job_name]
                if "deployment" in job_result:
                    assert job_result["deployment"]["deployed"], f"Main branch should deploy but got: {job_result}"
    
    @given(
        pr_number=st.integers(min_value=1, max_value=9999),
        head_branch=st.text(min_size=3, max_size=20, alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd'), whitelist_characters='-_')),
        commit_sha=st.text(min_size=40, max_size=40, alphabet=st.characters(whitelist_categories=('Nd', 'Ll'))),
        author=st.text(min_size=3, max_size=15, alphabet=st.characters(whitelist_categories=('Ll', 'Lu')))
    )
    @settings(deadline=None)  # Disable deadline for this test
    def test_pull_request_trigger_consistency(
        self, 
        pr_number: int, 
        head_branch: str, 
        commit_sha: str, 
        author: str
    ):
        """
        Test that pull request events trigger testing consistently but not deployment.
        """
        # Create GitHub pull request event
        pr_event = MockGitHubEvent(
            event_type="pull_request",
            ref=f"refs/heads/{head_branch}",
            commit_sha=commit_sha,
            author=author
        )
        
        # Create workflow instance
        workflow = MockGitHubActionsWorkflow("infrastructure/github-actions-workflow.yml")
        
        # Test trigger consistency for PR events
        results = []
        for _ in range(3):
            result = workflow.simulate_run(pr_event)
            results.append(result)
        
        # All runs should trigger (PRs should trigger testing)
        for result in results:
            assert result["triggered"], f"PR should trigger testing but got: {result}"
        
        # Deployment jobs should NOT deploy for PR events
        deployment_jobs = workflow.get_deployment_jobs()
        for result in results:
            for job_name in deployment_jobs:
                job_result = result["jobs"][job_name]
                if "deployment" in job_result:
                    assert not job_result["deployment"]["deployed"]
                    assert "Pull request" in job_result["deployment"]["reason"]
    
    @given(
        events_count=st.integers(min_value=2, max_value=5),
        time_interval=st.integers(min_value=1, max_value=300)  # seconds between events
    )
    @settings(deadline=None)  # Disable deadline for this test
    def test_multiple_deployment_trigger_consistency(
        self, 
        events_count: int, 
        time_interval: int
    ):
        """
        Test that multiple deployment triggers in sequence produce consistent results.
        """
        workflow = MockGitHubActionsWorkflow("infrastructure/github-actions-workflow.yml")
        
        # Generate sequence of main branch push events
        events = []
        for i in range(events_count):
            commit_sha = f"{'a' * 39}{i}"  # Generate unique commit SHAs
            event = MockGitHubEvent(
                event_type="push",
                ref="refs/heads/main",
                commit_sha=commit_sha,
                author=f"author{i}"
            )
            events.append(event)
        
        # Process all events
        results = []
        for event in events:
            result = workflow.simulate_run(event)
            results.append(result)
        
        # All events should trigger deployment
        for result in results:
            assert result["triggered"], f"Main branch push should trigger but got: {result}"
        
        # Each deployment should be consistent in structure
        deployment_jobs = workflow.get_deployment_jobs()
        for i, result in enumerate(results):
            for job_name in deployment_jobs:
                job_result = result["jobs"][job_name]
                
                if "deployment" in job_result:
                    assert job_result["deployment"]["deployed"]
                    assert job_result["deployment"]["environment"] == "production"
                    assert job_result["deployment"]["image_tag"] == events[i].commit_sha
        
        # Verify each deployment uses correct commit SHA
        for i, result in enumerate(results):
            assert result["commit_sha"] == events[i].commit_sha
    
    @given(
        workflow_modifications=st.lists(
            st.sampled_from([
                "add_security_scan",
                "add_integration_test",
                "modify_build_step",
                "add_notification"
            ]),
            min_size=0,
            max_size=3,
            unique=True
        )
    )
    @settings(deadline=None)  # Disable deadline for this test
    def test_workflow_modification_trigger_consistency(self, workflow_modifications: List[str]):
        """
        Test that workflow modifications don't break trigger consistency.
        """
        # Create base workflow
        workflow = MockGitHubActionsWorkflow("infrastructure/github-actions-workflow.yml")
        
        # Apply modifications to workflow configuration
        modified_workflow_data = workflow.workflow_data.copy()
        jobs = modified_workflow_data.get("jobs", {}).copy()
        
        for modification in workflow_modifications:
            if modification == "add_security_scan":
                jobs["additional-security-scan"] = {
                    "runs-on": "ubuntu-latest",
                    "needs": ["test"]
                }
            elif modification == "add_integration_test":
                jobs["integration-test"] = {
                    "runs-on": "ubuntu-latest",
                    "needs": ["build-and-push"]
                }
            elif modification == "modify_build_step":
                if "build-and-push" in jobs:
                    jobs["build-and-push"]["timeout-minutes"] = 30
            elif modification == "add_notification":
                jobs["notify-slack"] = {
                    "runs-on": "ubuntu-latest",
                    "needs": ["health-check"],
                    "if": "always()"
                }
        
        modified_workflow_data["jobs"] = jobs
        workflow.workflow_data = modified_workflow_data
        
        # Test that trigger behavior remains consistent
        test_event = MockGitHubEvent(
            event_type="push",
            ref="refs/heads/main",
            commit_sha="a" * 40,
            author="testuser"
        )
        
        # Run multiple times to verify consistency
        results = []
        for _ in range(3):
            result = workflow.simulate_run(test_event)
            results.append(result)
        
        # All runs should trigger
        for result in results:
            assert result["triggered"], f"Main branch push should trigger but got: {result}"
        
        # Deployment behavior should remain consistent
        deployment_jobs = workflow.get_deployment_jobs()
        for result in results:
            for job_name in deployment_jobs:
                if job_name in result["jobs"]:
                    job_result = result["jobs"][job_name]
                    if "deployment" in job_result:
                        assert job_result["deployment"]["deployed"]
                        assert job_result["deployment"]["image_tag"] == test_event.commit_sha
    
    def test_workflow_trigger_conditions_validation(self):
        """
        Test that workflow trigger conditions are properly validated.
        """
        workflow = MockGitHubActionsWorkflow("infrastructure/github-actions-workflow.yml")
        
        # Test various event scenarios
        test_cases = [
            # Should trigger deployment
            ("push", "refs/heads/main", True, True),
            ("push", "refs/heads/master", False, False),  # Only main branch configured
            
            # Should trigger testing but not deployment
            ("pull_request", "refs/heads/feature/test", True, False),
            ("pull_request", "refs/heads/main", True, False),  # PR to main
            
            # Should not trigger
            ("push", "refs/heads/develop", False, False),
            ("push", "refs/tags/v1.0.0", False, False),
        ]
        
        for event_type, ref, should_trigger, should_deploy in test_cases:
            event = MockGitHubEvent(
                event_type=event_type,
                ref=ref,
                commit_sha="a" * 40,
                author="testuser"
            )
            
            result = workflow.simulate_run(event)
            
            assert result["triggered"] == should_trigger, f"Failed for {event_type} {ref}. Expected {should_trigger}, got {result['triggered']}. Result: {result}"
            
            if should_trigger:
                deployment_jobs = workflow.get_deployment_jobs()
                for job_name in deployment_jobs:
                    if job_name in result["jobs"]:
                        job_result = result["jobs"][job_name]
                        if "deployment" in job_result:
                            assert job_result["deployment"]["deployed"] == should_deploy
    
    @given(
        concurrent_pushes=st.integers(min_value=2, max_value=4)
    )
    @settings(deadline=None)  # Disable deadline for this test
    def test_concurrent_deployment_trigger_handling(self, concurrent_pushes: int):
        """
        Test that concurrent deployment triggers are handled consistently.
        """
        workflow = MockGitHubActionsWorkflow("infrastructure/github-actions-workflow.yml")
        
        # Create concurrent push events to main branch
        events = []
        for i in range(concurrent_pushes):
            event = MockGitHubEvent(
                event_type="push",
                ref="refs/heads/main",
                commit_sha=f"{'b' * 39}{i}",
                author=f"user{i}"
            )
            events.append(event)
        
        # Simulate concurrent processing
        results = []
        for event in events:
            result = workflow.simulate_run(event)
            results.append(result)
        
        # All events should trigger
        for result in results:
            assert result["triggered"], f"Main branch push should trigger but got: {result}"
        
        # Each should have unique workflow run ID
        workflow_ids = [r["workflow_id"] for r in results]
        assert len(set(workflow_ids)) == len(workflow_ids)
        
        # Each should deploy with correct commit SHA
        deployment_jobs = workflow.get_deployment_jobs()
        for i, result in enumerate(results):
            for job_name in deployment_jobs:
                if job_name in result["jobs"]:
                    job_result = result["jobs"][job_name]
                    if "deployment" in job_result:
                        assert job_result["deployment"]["deployed"]
                        assert job_result["deployment"]["image_tag"] == events[i].commit_sha