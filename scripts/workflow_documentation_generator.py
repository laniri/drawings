"""
Workflow Documentation Generator

Generates BPMN 2.0 compliant workflow documentation including user journeys,
technical processes, integration flows, and error handling workflows.
"""

import os
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import xml.etree.ElementTree as ET
from xml.dom import minidom


class WorkflowType(Enum):
    """Types of workflows that can be documented."""
    USER_JOURNEY = "user_journey"
    TECHNICAL_PROCESS = "technical_process"
    INTEGRATION_FLOW = "integration_flow"
    ERROR_FLOW = "error_flow"
    ML_PIPELINE = "ml_pipeline"
    DATA_PROCESSING = "data_processing"


@dataclass
class BPMNElement:
    """Represents a BPMN 2.0 element."""
    element_id: str
    element_type: str
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    connections: List[str] = field(default_factory=list)
    position: Optional[Tuple[int, int]] = None


@dataclass
class WorkflowDiagram:
    """Represents a complete workflow diagram."""
    workflow_id: str
    name: str
    workflow_type: str
    elements: List[BPMNElement] = field(default_factory=list)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Type-specific attributes
    user_actions: List[str] = field(default_factory=list)
    system_responses: List[str] = field(default_factory=list)
    process_steps: List[str] = field(default_factory=list)
    external_systems: List[str] = field(default_factory=list)
    error_conditions: List[str] = field(default_factory=list)


@dataclass
class ProcessStep:
    """Represents a step in a workflow process."""
    step_id: str
    name: str
    step_type: str
    description: str = ""
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)


@dataclass
class UserJourney:
    """Represents a user journey workflow."""
    journey_id: str
    name: str
    user_type: str
    steps: List[ProcessStep] = field(default_factory=list)
    touchpoints: List[str] = field(default_factory=list)
    pain_points: List[str] = field(default_factory=list)


@dataclass
class TechnicalProcess:
    """Represents a technical process workflow."""
    process_id: str
    name: str
    service_name: str
    steps: List[ProcessStep] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    data_flows: List[str] = field(default_factory=list)


@dataclass
class IntegrationFlow:
    """Represents an integration workflow."""
    flow_id: str
    name: str
    source_system: str
    target_system: str
    steps: List[ProcessStep] = field(default_factory=list)
    protocols: List[str] = field(default_factory=list)
    data_formats: List[str] = field(default_factory=list)


@dataclass
class ErrorFlow:
    """Represents an error handling workflow."""
    flow_id: str
    name: str
    error_type: str
    steps: List[ProcessStep] = field(default_factory=list)
    recovery_actions: List[str] = field(default_factory=list)
    escalation_paths: List[str] = field(default_factory=list)


class WorkflowGenerator:
    """
    Generates comprehensive workflow documentation with BPMN 2.0 compliance.
    
    Extracts workflow patterns from code and configuration to create
    user journeys, technical processes, integration flows, and error handling workflows.
    """
    
    def __init__(self, project_path: str):
        """Initialize the workflow generator."""
        self.project_path = project_path
        self.workflow_types = [wt.value for wt in WorkflowType]
        self.bpmn_elements = [
            "startEvent", "endEvent", "task", "serviceTask", "userTask",
            "gateway", "exclusiveGateway", "parallelGateway", "sequenceFlow",
            "boundaryEvent", "errorEvent", "timerEvent", "messageEvent"
        ]
        self.workflows: Dict[str, WorkflowDiagram] = {}
        
    def generate_all_workflows(self) -> Dict[str, WorkflowDiagram]:
        """Generate all workflow documentation."""
        self.workflows = {}
        
        # Extract workflows from different sources
        self._extract_user_journeys()
        self._extract_technical_processes()
        self._extract_integration_flows()
        self._extract_error_flows()
        self._extract_ml_pipelines()
        
        return self.workflows
    
    def _extract_user_journeys(self) -> None:
        """Extract user journey workflows from frontend code."""
        frontend_path = Path(self.project_path) / "frontend"
        if not frontend_path.exists():
            return
            
        # Analyze React components for user flows
        for tsx_file in frontend_path.rglob("*.tsx"):
            try:
                content = tsx_file.read_text(encoding='utf-8')
                journey = self._analyze_react_component(tsx_file, content)
                if journey:
                    self.workflows[journey.workflow_id] = journey
            except Exception:
                continue
    
    def _extract_technical_processes(self) -> None:
        """Extract technical process workflows from service code."""
        app_path = Path(self.project_path) / "app"
        if not app_path.exists():
            return
            
        # Analyze service files for technical processes
        services_path = app_path / "services"
        if services_path.exists():
            for py_file in services_path.glob("*.py"):
                try:
                    content = py_file.read_text(encoding='utf-8')
                    process = self._analyze_service_file(py_file, content)
                    if process:
                        self.workflows[process.workflow_id] = process
                except Exception:
                    continue
    
    def _extract_integration_flows(self) -> None:
        """Extract integration workflows from API and external service calls."""
        app_path = Path(self.project_path) / "app"
        if not app_path.exists():
            return
            
        # Analyze API endpoints for integration patterns
        api_path = app_path / "api"
        if api_path.exists():
            for py_file in api_path.rglob("*.py"):
                try:
                    content = py_file.read_text(encoding='utf-8')
                    flows = self._analyze_api_integrations(py_file, content)
                    for flow in flows:
                        self.workflows[flow.workflow_id] = flow
                except Exception:
                    continue
    
    def _extract_error_flows(self) -> None:
        """Extract error handling workflows from exception handling code."""
        app_path = Path(self.project_path) / "app"
        if not app_path.exists():
            return
            
        # Analyze exception handling patterns
        for py_file in app_path.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                error_flows = self._analyze_error_handling(py_file, content)
                for flow in error_flows:
                    self.workflows[flow.workflow_id] = flow
            except Exception:
                continue
    
    def _extract_ml_pipelines(self) -> None:
        """Extract ML pipeline workflows from training and inference code."""
        # Look for ML pipeline files
        ml_files = [
            "train_models.py", "train_models_offline.py", "train_models_direct.py"
        ]
        
        for ml_file in ml_files:
            file_path = Path(self.project_path) / ml_file
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding='utf-8')
                    pipeline = self._analyze_ml_pipeline(file_path, content)
                    if pipeline:
                        self.workflows[pipeline.workflow_id] = pipeline
                except Exception:
                    continue
    
    def _analyze_react_component(self, file_path: Path, content: str) -> Optional[WorkflowDiagram]:
        """Analyze React component for user journey patterns."""
        component_name = file_path.stem
        
        # Extract user interactions and state changes
        user_actions = self._extract_user_actions(content)
        system_responses = self._extract_system_responses(content)
        
        if not user_actions and not system_responses:
            return None
        
        # Create BPMN elements for user journey
        elements = []
        element_id = 0
        
        # Start event
        elements.append(BPMNElement(
            element_id=f"start_{element_id}",
            element_type="startEvent",
            name="User Starts Journey"
        ))
        element_id += 1
        
        # User actions as user tasks
        for action in user_actions:
            elements.append(BPMNElement(
                element_id=f"user_task_{element_id}",
                element_type="userTask",
                name=action,
                properties={"actor": "user"}
            ))
            element_id += 1
        
        # System responses as service tasks
        for response in system_responses:
            elements.append(BPMNElement(
                element_id=f"system_task_{element_id}",
                element_type="serviceTask",
                name=response,
                properties={"actor": "system"}
            ))
            element_id += 1
        
        # End event
        elements.append(BPMNElement(
            element_id=f"end_{element_id}",
            element_type="endEvent",
            name="Journey Complete"
        ))
        
        return WorkflowDiagram(
            workflow_id=f"user_journey_{component_name.lower()}",
            name=f"User Journey - {component_name}",
            workflow_type=WorkflowType.USER_JOURNEY.value,
            elements=elements,
            description=f"User journey workflow for {component_name} component",
            user_actions=user_actions,
            system_responses=system_responses
        )
    
    def _analyze_service_file(self, file_path: Path, content: str) -> Optional[WorkflowDiagram]:
        """Analyze service file for technical process patterns."""
        service_name = file_path.stem
        
        # Parse Python AST to extract methods and their flow
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return None
        
        process_steps = []
        methods = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                methods.append(node.name)
                # Extract method calls and data flow
                for child in ast.walk(node):
                    if isinstance(child, ast.Call) and hasattr(child.func, 'attr'):
                        process_steps.append(child.func.attr)
        
        if not methods:
            return None
        
        # Create BPMN elements for technical process
        elements = []
        element_id = 0
        
        # Start event
        elements.append(BPMNElement(
            element_id=f"start_{element_id}",
            element_type="startEvent",
            name="Process Start"
        ))
        element_id += 1
        
        # Methods as service tasks
        for method in methods[:5]:  # Limit to first 5 methods
            elements.append(BPMNElement(
                element_id=f"service_task_{element_id}",
                element_type="serviceTask",
                name=method.replace('_', ' ').title(),
                properties={"service": service_name, "method": method}
            ))
            element_id += 1
        
        # End event
        elements.append(BPMNElement(
            element_id=f"end_{element_id}",
            element_type="endEvent",
            name="Process Complete"
        ))
        
        return WorkflowDiagram(
            workflow_id=f"technical_process_{service_name}",
            name=f"Technical Process - {service_name.replace('_', ' ').title()}",
            workflow_type=WorkflowType.TECHNICAL_PROCESS.value,
            elements=elements,
            description=f"Technical process workflow for {service_name} service",
            process_steps=process_steps
        )
    
    def _analyze_api_integrations(self, file_path: Path, content: str) -> List[WorkflowDiagram]:
        """Analyze API file for integration flow patterns."""
        flows = []
        
        # Look for external API calls and integrations
        external_calls = re.findall(r'requests\.(get|post|put|delete)\(', content)
        aws_calls = re.findall(r'boto3\.|sagemaker\.|s3\.|lambda_', content)
        
        if external_calls or aws_calls:
            elements = []
            element_id = 0
            
            # Start event
            elements.append(BPMNElement(
                element_id=f"start_{element_id}",
                element_type="startEvent",
                name="Integration Start"
            ))
            element_id += 1
            
            # External calls as send/receive tasks
            for call in external_calls[:3]:  # Limit to first 3
                elements.append(BPMNElement(
                    element_id=f"send_task_{element_id}",
                    element_type="sendTask",
                    name=f"HTTP {call[0].upper()} Request",
                    properties={"protocol": "HTTP", "method": call[0].upper()}
                ))
                element_id += 1
                
                elements.append(BPMNElement(
                    element_id=f"receive_task_{element_id}",
                    element_type="receiveTask",
                    name=f"Receive {call[0].upper()} Response",
                    properties={"protocol": "HTTP"}
                ))
                element_id += 1
            
            # AWS calls as service tasks
            for aws_call in aws_calls[:2]:  # Limit to first 2
                elements.append(BPMNElement(
                    element_id=f"aws_task_{element_id}",
                    element_type="serviceTask",
                    name=f"AWS {aws_call.replace('_', ' ').title()}",
                    properties={"provider": "AWS", "service": aws_call}
                ))
                element_id += 1
            
            # End event
            elements.append(BPMNElement(
                element_id=f"end_{element_id}",
                element_type="endEvent",
                name="Integration Complete"
            ))
            
            flow = WorkflowDiagram(
                workflow_id=f"integration_flow_{file_path.stem}",
                name=f"Integration Flow - {file_path.stem.replace('_', ' ').title()}",
                workflow_type=WorkflowType.INTEGRATION_FLOW.value,
                elements=elements,
                description=f"Integration workflow for {file_path.stem}",
                external_systems=["HTTP APIs", "AWS Services"]
            )
            flows.append(flow)
        
        return flows
    
    def _analyze_error_handling(self, file_path: Path, content: str) -> List[WorkflowDiagram]:
        """Analyze file for error handling patterns."""
        flows = []
        
        # Look for exception handling patterns
        try_blocks = re.findall(r'try:(.*?)except\s+(\w+)', content, re.DOTALL)
        raise_statements = re.findall(r'raise\s+(\w+)', content)
        
        if try_blocks or raise_statements:
            elements = []
            element_id = 0
            
            # Start event
            elements.append(BPMNElement(
                element_id=f"start_{element_id}",
                element_type="startEvent",
                name="Error Condition Detected"
            ))
            element_id += 1
            
            # Error events
            for _, exception_type in try_blocks[:3]:  # Limit to first 3
                elements.append(BPMNElement(
                    element_id=f"error_event_{element_id}",
                    element_type="errorEvent",
                    name=f"{exception_type} Error",
                    properties={"error_type": exception_type}
                ))
                element_id += 1
                
                elements.append(BPMNElement(
                    element_id=f"recovery_task_{element_id}",
                    element_type="task",
                    name=f"Handle {exception_type}",
                    properties={"recovery": True}
                ))
                element_id += 1
            
            # End event
            elements.append(BPMNElement(
                element_id=f"end_{element_id}",
                element_type="endEvent",
                name="Error Resolved"
            ))
            
            error_conditions = [exc_type for _, exc_type in try_blocks] + raise_statements
            
            flow = WorkflowDiagram(
                workflow_id=f"error_flow_{file_path.stem}",
                name=f"Error Flow - {file_path.stem.replace('_', ' ').title()}",
                workflow_type=WorkflowType.ERROR_FLOW.value,
                elements=elements,
                description=f"Error handling workflow for {file_path.stem}",
                error_conditions=error_conditions
            )
            flows.append(flow)
        
        return flows
    
    def _analyze_ml_pipeline(self, file_path: Path, content: str) -> Optional[WorkflowDiagram]:
        """Analyze ML pipeline file for workflow patterns."""
        pipeline_name = file_path.stem
        
        # Look for ML pipeline steps
        ml_steps = []
        if 'load_data' in content or 'prepare_data' in content:
            ml_steps.append("Data Loading")
        if 'preprocess' in content or 'transform' in content:
            ml_steps.append("Data Preprocessing")
        if 'train' in content or 'fit' in content:
            ml_steps.append("Model Training")
        if 'evaluate' in content or 'validate' in content:
            ml_steps.append("Model Evaluation")
        if 'save' in content or 'export' in content:
            ml_steps.append("Model Saving")
        
        if not ml_steps:
            return None
        
        # Create BPMN elements for ML pipeline
        elements = []
        element_id = 0
        
        # Start event
        elements.append(BPMNElement(
            element_id=f"start_{element_id}",
            element_type="startEvent",
            name="ML Pipeline Start"
        ))
        element_id += 1
        
        # ML steps as service tasks
        for step in ml_steps:
            elements.append(BPMNElement(
                element_id=f"ml_task_{element_id}",
                element_type="serviceTask",
                name=step,
                properties={"pipeline": pipeline_name, "step_type": "ml"}
            ))
            element_id += 1
        
        # End event
        elements.append(BPMNElement(
            element_id=f"end_{element_id}",
            element_type="endEvent",
            name="ML Pipeline Complete"
        ))
        
        return WorkflowDiagram(
            workflow_id=f"ml_pipeline_{pipeline_name}",
            name=f"ML Pipeline - {pipeline_name.replace('_', ' ').title()}",
            workflow_type=WorkflowType.ML_PIPELINE.value,
            elements=elements,
            description=f"Machine learning pipeline workflow for {pipeline_name}",
            process_steps=ml_steps
        )
    
    def _extract_user_actions(self, content: str) -> List[str]:
        """Extract user actions from React component."""
        actions = []
        
        # Look for event handlers and user interactions
        onclick_patterns = re.findall(r'onClick=\{([^}]+)\}', content)
        onsubmit_patterns = re.findall(r'onSubmit=\{([^}]+)\}', content)
        onchange_patterns = re.findall(r'onChange=\{([^}]+)\}', content)
        
        for pattern in onclick_patterns:
            actions.append(f"Click {pattern.replace('handle', '').replace('()', '')}")
        for pattern in onsubmit_patterns:
            actions.append(f"Submit {pattern.replace('handle', '').replace('()', '')}")
        for pattern in onchange_patterns:
            actions.append(f"Change {pattern.replace('handle', '').replace('()', '')}")
        
        return actions[:5]  # Limit to first 5 actions
    
    def _extract_system_responses(self, content: str) -> List[str]:
        """Extract system responses from React component."""
        responses = []
        
        # Look for API calls and state updates
        api_calls = re.findall(r'fetch\([\'"]([^\'"]+)[\'"]', content)
        axios_calls = re.findall(r'axios\.(get|post|put|delete)\([\'"]([^\'"]+)[\'"]', content)
        state_updates = re.findall(r'set\w+\(', content)
        
        for call in api_calls:
            responses.append(f"API Call to {call}")
        for method, url in axios_calls:
            responses.append(f"{method.upper()} Request to {url}")
        for update in state_updates:
            responses.append(f"Update {update.replace('set', '').replace('(', '')}")
        
        return responses[:5]  # Limit to first 5 responses
    
    def create_workflow_from_data(self, workflow_data: Dict[str, Any], workflow_type: str) -> WorkflowDiagram:
        """Create a workflow diagram from provided data."""
        workflow_id = f"workflow_{len(self.workflows)}"
        name = workflow_data.get('name', f'Workflow {workflow_id}')
        
        # Create basic elements
        elements = []
        element_id = 0
        
        # Start event
        elements.append(BPMNElement(
            element_id=f"start_{element_id}",
            element_type="startEvent",
            name="Start"
        ))
        element_id += 1
        
        # Create tasks from data
        for key, value in workflow_data.items():
            if isinstance(value, list):
                for item in value[:3]:  # Limit to first 3 items
                    elements.append(BPMNElement(
                        element_id=f"task_{element_id}",
                        element_type="task",
                        name=str(item)[:30],  # Limit name length
                        properties={"source": key}
                    ))
                    element_id += 1
            elif isinstance(value, str) and len(value) > 0:
                elements.append(BPMNElement(
                    element_id=f"task_{element_id}",
                    element_type="task",
                    name=value[:30],  # Limit name length
                    properties={"source": key}
                ))
                element_id += 1
        
        # End event
        elements.append(BPMNElement(
            element_id=f"end_{element_id}",
            element_type="endEvent",
            name="End"
        ))
        
        return WorkflowDiagram(
            workflow_id=workflow_id,
            name=name,
            workflow_type=workflow_type,
            elements=elements,
            description=f"Generated workflow from data"
        )
    
    def organize_workflows(self, workflows: List[WorkflowDiagram]) -> Dict[str, List[WorkflowDiagram]]:
        """Organize workflows by type and complexity."""
        organized = {}
        
        for workflow in workflows:
            wf_type = workflow.workflow_type
            if wf_type not in organized:
                organized[wf_type] = []
            organized[wf_type].append(workflow)
        
        # Sort by complexity (number of elements)
        for wf_type in organized:
            organized[wf_type].sort(key=lambda w: len(w.elements), reverse=True)
        
        return organized
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get workflow generator configuration."""
        return {
            'supported_workflow_types': self.workflow_types,
            'bpmn_version': '2.0',
            'supported_elements': self.bpmn_elements,
            'project_path': self.project_path
        }
    
    def generate_bpmn_xml(self, workflow: WorkflowDiagram) -> str:
        """Generate BPMN 2.0 XML for a workflow diagram."""
        # Create XML structure
        root = ET.Element("bpmn:definitions")
        root.set("xmlns:bpmn", "http://www.omg.org/spec/BPMN/20100524/MODEL")
        root.set("xmlns:bpmndi", "http://www.omg.org/spec/BPMN/20100524/DI")
        root.set("xmlns:dc", "http://www.omg.org/spec/DD/20100524/DC")
        root.set("xmlns:di", "http://www.omg.org/spec/DD/20100524/DI")
        root.set("id", f"Definitions_{self._sanitize_xml_id(workflow.workflow_id)}")
        root.set("targetNamespace", "http://bpmn.io/schema/bpmn")
        
        # Create process
        process = ET.SubElement(root, "bpmn:process")
        process.set("id", self._sanitize_xml_id(workflow.workflow_id))
        process.set("name", self._sanitize_xml_text(workflow.name))
        process.set("isExecutable", "false")
        
        # Add elements
        for element in workflow.elements:
            elem = ET.SubElement(process, f"bpmn:{element.element_type}")
            elem.set("id", self._sanitize_xml_id(element.element_id))
            elem.set("name", self._sanitize_xml_text(element.name))
            
            # Add properties if any
            for prop_key, prop_value in element.properties.items():
                safe_key = self._sanitize_xml_id(prop_key)
                safe_value = self._sanitize_xml_text(str(prop_value))
                elem.set(safe_key, safe_value)
        
        # Add sequence flows between elements
        for i in range(len(workflow.elements) - 1):
            current = workflow.elements[i]
            next_elem = workflow.elements[i + 1]
            
            flow = ET.SubElement(process, "bpmn:sequenceFlow")
            flow.set("id", f"flow_{i}")
            flow.set("sourceRef", self._sanitize_xml_id(current.element_id))
            flow.set("targetRef", self._sanitize_xml_id(next_elem.element_id))
        
        # Convert to string with proper formatting
        rough_string = ET.tostring(root, encoding='unicode')
        try:
            reparsed = minidom.parseString(rough_string)
            return reparsed.toprettyxml(indent="  ")
        except Exception:
            # Fallback to raw XML if parsing fails
            return rough_string
    
    def _sanitize_xml_id(self, text: str) -> str:
        """Sanitize text for use as XML ID (must start with letter/underscore, contain only valid chars)."""
        if not text:
            return "element_0"
        
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_.-]', '_', str(text))
        
        # Ensure it starts with a letter or underscore
        if not sanitized[0].isalpha() and sanitized[0] != '_':
            sanitized = f"element_{sanitized}"
        
        return sanitized
    
    def _sanitize_xml_text(self, text: str) -> str:
        """Sanitize text for use as XML text content."""
        if not text:
            return ""
        
        # Replace control characters and other problematic characters
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', str(text))
        
        # Escape XML special characters
        sanitized = sanitized.replace('&', '&amp;')
        sanitized = sanitized.replace('<', '&lt;')
        sanitized = sanitized.replace('>', '&gt;')
        sanitized = sanitized.replace('"', '&quot;')
        sanitized = sanitized.replace("'", '&apos;')
        
        return sanitized
    
    def generate_workflow_documentation(self, output_dir: str) -> Dict[str, str]:
        """Generate comprehensive workflow documentation files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = {}
        
        for workflow_id, workflow in self.workflows.items():
            # Generate Markdown documentation
            md_content = self._generate_workflow_markdown(workflow)
            md_file = output_path / f"{workflow_id}.md"
            md_file.write_text(md_content, encoding='utf-8')
            generated_files[f"{workflow_id}.md"] = str(md_file)
            
            # Generate BPMN XML
            bpmn_xml = self.generate_bpmn_xml(workflow)
            bpmn_file = output_path / f"{workflow_id}.bpmn"
            bpmn_file.write_text(bpmn_xml, encoding='utf-8')
            generated_files[f"{workflow_id}.bpmn"] = str(bpmn_file)
        
        return generated_files
    
    def _generate_workflow_markdown(self, workflow: WorkflowDiagram) -> str:
        """Generate Markdown documentation for a workflow."""
        content = f"""# {workflow.name}

**Workflow ID**: `{workflow.workflow_id}`
**Type**: {workflow.workflow_type.replace('_', ' ').title()}
**Last Updated**: {Path().ctime() if hasattr(Path(), 'ctime') else 'Unknown'}

## Overview

{workflow.description}

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
"""
        
        for element in workflow.elements:
            props = ", ".join([f"{k}={v}" for k, v in element.properties.items()])
            content += f"| {element.element_id} | {element.element_type} | {element.name} | {props} |\n"
        
        # Add type-specific sections
        if workflow.workflow_type == WorkflowType.USER_JOURNEY.value:
            content += f"""
## User Actions

{chr(10).join([f"- {action}" for action in workflow.user_actions])}

## System Responses

{chr(10).join([f"- {response}" for response in workflow.system_responses])}
"""
        
        elif workflow.workflow_type == WorkflowType.TECHNICAL_PROCESS.value:
            content += f"""
## Process Steps

{chr(10).join([f"- {step}" for step in workflow.process_steps])}
"""
        
        elif workflow.workflow_type == WorkflowType.INTEGRATION_FLOW.value:
            content += f"""
## External Systems

{chr(10).join([f"- {system}" for system in workflow.external_systems])}
"""
        
        elif workflow.workflow_type == WorkflowType.ERROR_FLOW.value:
            content += f"""
## Error Conditions

{chr(10).join([f"- {condition}" for condition in workflow.error_conditions])}
"""
        
        content += f"""
## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `{workflow.workflow_id}.bpmn`.

## Metadata

- **Elements Count**: {len(workflow.elements)}
- **Workflow Type**: {workflow.workflow_type}
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
"""
        
        return content