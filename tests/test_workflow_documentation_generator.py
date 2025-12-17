"""
Property-based tests for workflow documentation generation.

Tests comprehensive workflow documentation generation including BPMN 2.0 compliance,
user journey extraction, technical process flows, and integration workflows.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from pathlib import Path
import tempfile
import os
import sys
from typing import Dict, List, Any, Optional
import json
import re

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.workflow_documentation_generator import (
    WorkflowGenerator,
    WorkflowType,
    ProcessStep,
    WorkflowDiagram,
    BPMNElement,
    UserJourney,
    TechnicalProcess,
    IntegrationFlow,
    ErrorFlow
)


class TestWorkflowDocumentationGenerator:
    """Property-based tests for workflow documentation generation."""

    @given(
        service_files=st.lists(
            st.just("""
class TestService:
    def process_data(self):
        return "processed"
    
    def validate_input(self):
        return True
"""),
            min_size=1,
            max_size=3
        ),
        frontend_files=st.lists(
            st.just("""
function TestComponent() {
    const handleClick = () => {};
    const handleSubmit = () => {};
    return <div onClick={handleClick}>Test</div>;
}
"""),
            min_size=1,
            max_size=2
        ),
        config_data=st.dictionaries(
            st.text(min_size=1, max_size=20, alphabet='abcdefghijklmnopqrstuvwxyz'),
            st.one_of(st.text(min_size=1, max_size=50), st.integers(min_value=0, max_value=1000), st.booleans()),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=10, deadline=10000)
    def test_comprehensive_workflow_documentation_generation(
        self, service_files: List[str], frontend_files: List[str], config_data: Dict[str, Any]
    ):
        """
        Property 3: Complete Workflow Documentation Generation
        Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5
        
        Tests that workflow documentation generation produces comprehensive,
        BPMN 2.0 compliant workflow diagrams for all process types.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test project structure
            app_dir = Path(temp_dir) / "app"
            frontend_dir = Path(temp_dir) / "frontend"
            app_dir.mkdir(parents=True)
            frontend_dir.mkdir(parents=True)
            
            # Write service files
            for i, content in enumerate(service_files):
                service_file = app_dir / f"service_{i}.py"
                service_file.write_text(content)
            
            # Write frontend files
            for i, content in enumerate(frontend_files):
                frontend_file = frontend_dir / f"component_{i}.tsx"
                frontend_file.write_text(content)
            
            # Write config file
            config_file = Path(temp_dir) / "config.json"
            config_file.write_text(json.dumps(config_data))
            
            # Initialize workflow generator
            generator = WorkflowGenerator(str(temp_dir))
            
            # Generate workflow documentation
            workflows = generator.generate_all_workflows()
            
            # Property 1: BPMN 2.0 Compliance (Requirement 3.1)
            assert isinstance(workflows, dict), "Workflows should be returned as dictionary"
            assert len(workflows) > 0, "Should generate at least one workflow"
            
            for workflow_name, workflow in workflows.items():
                assert isinstance(workflow, WorkflowDiagram), f"Workflow {workflow_name} should be WorkflowDiagram instance"
                assert workflow.workflow_type in [wt.value for wt in WorkflowType], f"Invalid workflow type: {workflow.workflow_type}"
                assert len(workflow.elements) > 0, f"Workflow {workflow_name} should have BPMN elements"
                
                # Validate BPMN 2.0 structure
                start_events = [e for e in workflow.elements if e.element_type == "startEvent"]
                end_events = [e for e in workflow.elements if e.element_type == "endEvent"]
                assert len(start_events) >= 1, f"Workflow {workflow_name} should have at least one start event"
                assert len(end_events) >= 1, f"Workflow {workflow_name} should have at least one end event"
            
            # Property 2: User Journey Documentation (Requirement 3.1)
            user_journeys = [w for w in workflows.values() if w.workflow_type == WorkflowType.USER_JOURNEY.value]
            if user_journeys:
                for journey in user_journeys:
                    assert hasattr(journey, 'user_actions'), "User journey should have user actions"
                    assert hasattr(journey, 'system_responses'), "User journey should have system responses"
                    assert len(journey.elements) >= 3, "User journey should have multiple steps"
            
            # Property 3: Technical Process Documentation (Requirement 3.2)
            technical_processes = [w for w in workflows.values() if w.workflow_type == WorkflowType.TECHNICAL_PROCESS.value]
            if technical_processes:
                for process in technical_processes:
                    assert hasattr(process, 'process_steps'), "Technical process should have process steps"
                    assert any(e.element_type == "task" for e in process.elements), "Technical process should have tasks"
            
            # Property 4: Integration Flow Documentation (Requirement 3.3)
            integration_flows = [w for w in workflows.values() if w.workflow_type == WorkflowType.INTEGRATION_FLOW.value]
            if integration_flows:
                for flow in integration_flows:
                    assert hasattr(flow, 'external_systems'), "Integration flow should identify external systems"
                    assert any(e.element_type in ["serviceTask", "sendTask", "receiveTask"] for e in flow.elements), \
                        "Integration flow should have service interaction elements"
            
            # Property 5: Error Handling Documentation (Requirement 3.4)
            error_flows = [w for w in workflows.values() if w.workflow_type == WorkflowType.ERROR_FLOW.value]
            if error_flows:
                for flow in error_flows:
                    assert hasattr(flow, 'error_conditions'), "Error flow should define error conditions"
                    assert any(e.element_type in ["boundaryEvent", "errorEvent"] for e in flow.elements), \
                        "Error flow should have error handling elements"
            
            # Property 6: BPMN Diagram Generation (Requirement 3.5)
            for workflow in workflows.values():
                bpmn_xml = generator.generate_bpmn_xml(workflow)
                assert bpmn_xml is not None, "Should generate BPMN XML"
                assert isinstance(bpmn_xml, str), "BPMN XML should be string"
                assert "<?xml" in bpmn_xml, "Should be valid XML"
                assert "bpmn:" in bpmn_xml, "Should contain BPMN namespace"
                assert "process" in bpmn_xml, "Should contain process definition"
                
                # Validate XML structure
                assert bpmn_xml.count("<") == bpmn_xml.count(">"), "XML should be well-formed"
                assert "startEvent" in bpmn_xml, "Should contain start event"
                assert "endEvent" in bpmn_xml, "Should contain end event"

    @given(
        workflow_data=st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(
                st.text(min_size=1, max_size=100),
                st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=5)
            ),
            min_size=1,
            max_size=10
        ),
        workflow_type=st.sampled_from([wt.value for wt in WorkflowType])
    )
    @settings(max_examples=30, deadline=3000)
    def test_workflow_extraction_and_parsing(
        self, workflow_data: Dict[str, Any], workflow_type: str
    ):
        """Test workflow extraction from various sources and parsing accuracy."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = WorkflowGenerator(temp_dir)
            
            # Create workflow from data
            workflow = generator.create_workflow_from_data(workflow_data, workflow_type)
            
            # Validate workflow structure
            assert isinstance(workflow, WorkflowDiagram), "Should create WorkflowDiagram instance"
            assert workflow.workflow_type == workflow_type, "Should preserve workflow type"
            assert len(workflow.elements) > 0, "Should have workflow elements"
            
            # Validate element properties
            for element in workflow.elements:
                assert isinstance(element, BPMNElement), "Elements should be BPMNElement instances"
                assert element.element_id is not None, "Elements should have IDs"
                assert element.element_type is not None, "Elements should have types"
                assert len(element.element_id) > 0, "Element IDs should not be empty"
                assert len(element.element_type) > 0, "Element types should not be empty"

    @given(
        workflows=st.lists(
            st.builds(
                WorkflowDiagram,
                workflow_id=st.text(min_size=1, max_size=20),
                name=st.text(min_size=1, max_size=50),
                workflow_type=st.sampled_from([wt.value for wt in WorkflowType]),
                elements=st.lists(
                    st.builds(
                        BPMNElement,
                        element_id=st.text(min_size=1, max_size=20),
                        element_type=st.sampled_from([
                            "startEvent", "endEvent", "task", "serviceTask",
                            "userTask", "gateway", "sequenceFlow"
                        ]),
                        name=st.text(min_size=1, max_size=30),
                        properties=st.dictionaries(
                            st.text(min_size=1, max_size=10),
                            st.text(min_size=1, max_size=20),
                            min_size=0,
                            max_size=5
                        )
                    ),
                    min_size=1,
                    max_size=10
                )
            ),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=20, deadline=3000)
    def test_workflow_categorization_and_organization(self, workflows: List[WorkflowDiagram]):
        """Test workflow categorization and organization by type and complexity."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = WorkflowGenerator(temp_dir)
            
            # Organize workflows
            organized = generator.organize_workflows(workflows)
            
            # Validate organization structure
            assert isinstance(organized, dict), "Should return organized dictionary"
            
            # Check that all workflow types are represented
            workflow_types = set(w.workflow_type for w in workflows)
            for wf_type in workflow_types:
                assert wf_type in organized, f"Should include workflow type {wf_type}"
                assert len(organized[wf_type]) > 0, f"Should have workflows for type {wf_type}"
            
            # Validate workflow preservation
            total_organized = sum(len(wfs) for wfs in organized.values())
            assert total_organized == len(workflows), "Should preserve all workflows"
            
            # Validate workflow integrity
            for wf_type, wf_list in organized.items():
                for workflow in wf_list:
                    assert workflow.workflow_type == wf_type, "Workflow should match category"
                    assert len(workflow.elements) > 0, "Workflow should have elements"

    @given(
        project_path=st.text(min_size=1, max_size=20, alphabet='abcdefghijklmnopqrstuvwxyz0123456789_-')
    )
    @settings(max_examples=10, deadline=2000)
    def test_workflow_documentation_engine_initialization(self, project_path: str):
        """Test workflow documentation engine initialization and configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            full_path = os.path.join(temp_dir, project_path)
            os.makedirs(full_path, exist_ok=True)
            
            # Initialize generator
            generator = WorkflowGenerator(full_path)
            
            # Validate initialization
            assert generator.project_path == full_path, "Should store project path"
            assert hasattr(generator, 'workflow_types'), "Should have workflow types"
            assert hasattr(generator, 'bpmn_elements'), "Should have BPMN elements"
            
            # Validate configuration
            config = generator.get_configuration()
            assert isinstance(config, dict), "Configuration should be dictionary"
            assert 'supported_workflow_types' in config, "Should list supported workflow types"
            assert 'bpmn_version' in config, "Should specify BPMN version"
            assert config['bpmn_version'] == '2.0', "Should support BPMN 2.0"

    @given(
        bpmn_elements=st.lists(
            st.builds(
                BPMNElement,
                element_id=st.text(min_size=1, max_size=20, alphabet='abcdefghijklmnopqrstuvwxyz0123456789_'),
                element_type=st.sampled_from([
                    "startEvent", "endEvent", "task", "serviceTask",
                    "userTask", "gateway", "sequenceFlow", "boundaryEvent"
                ]),
                name=st.text(min_size=1, max_size=30, alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-'),
                properties=st.dictionaries(
                    st.text(min_size=1, max_size=10, alphabet='abcdefghijklmnopqrstuvwxyz'),
                    st.text(min_size=1, max_size=20, alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-'),
                    min_size=0,
                    max_size=3
                )
            ),
            min_size=2,
            max_size=5
        )
    )
    @settings(max_examples=20, deadline=3000)
    def test_bpmn_xml_generation_and_validation(self, bpmn_elements: List[BPMNElement]):
        """Test BPMN XML generation and validation for compliance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = WorkflowGenerator(temp_dir)
            
            # Create workflow with elements
            workflow = WorkflowDiagram(
                workflow_id="test_workflow",
                name="Test Workflow",
                workflow_type=WorkflowType.TECHNICAL_PROCESS.value,
                elements=bpmn_elements
            )
            
            # Generate BPMN XML
            bpmn_xml = generator.generate_bpmn_xml(workflow)
            
            # Validate XML structure
            assert isinstance(bpmn_xml, str), "Should generate string XML"
            assert len(bpmn_xml) > 0, "XML should not be empty"
            assert "<?xml" in bpmn_xml, "Should have XML declaration"
            assert "bpmn:" in bpmn_xml, "Should use BPMN namespace"
            
            # Validate BPMN 2.0 compliance
            assert 'xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL"' in bpmn_xml, \
                "Should use correct BPMN 2.0 namespace"
            assert "<bpmn:process" in bpmn_xml, "Should contain process definition"
            
            # Validate element inclusion
            for element in bpmn_elements:
                element_tag = f"<bpmn:{element.element_type}"
                assert element_tag in bpmn_xml, f"Should include element {element.element_type}"
                assert element.element_id in bpmn_xml, f"Should include element ID {element.element_id}"
            
            # Validate XML well-formedness
            open_tags = bpmn_xml.count("<")
            close_tags = bpmn_xml.count(">")
            assert open_tags == close_tags, "XML should be well-formed"