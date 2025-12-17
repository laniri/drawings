# {{WORKFLOW_NAME}} Workflow

## Overview

{{WORKFLOW_DESCRIPTION}}

**Workflow Type**: {{WORKFLOW_TYPE}}  
**BPMN Version**: 2.0  
**Process Owner**: {{PROCESS_OWNER}}

## Process Diagram

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryColor':'#ff6b6b', 'primaryTextColor':'#fff', 'primaryBorderColor':'#ff4757', 'lineColor':'#5f27cd', 'secondaryColor':'#00d2d3', 'tertiaryColor':'#ff9ff3'}}}%%
graph TD
    {{WORKFLOW_DIAGRAM}}
```

## BPMN Elements

### Start Events

{{#START_EVENTS}}
#### {{EVENT_NAME}}

- **Type**: {{EVENT_TYPE}}
- **Trigger**: {{EVENT_TRIGGER}}
- **Description**: {{EVENT_DESCRIPTION}}
{{/START_EVENTS}}

### Tasks

{{#TASKS}}
#### {{TASK_NAME}}

- **Type**: {{TASK_TYPE}}
- **Performer**: {{TASK_PERFORMER}}
- **Description**: {{TASK_DESCRIPTION}}
- **Input**: {{TASK_INPUT}}
- **Output**: {{TASK_OUTPUT}}
- **Duration**: {{TASK_DURATION}}

{{#TASK_PROPERTIES}}
**Properties**:
{{#PROPERTIES}}
- **{{PROPERTY_NAME}}**: {{PROPERTY_VALUE}}
{{/PROPERTIES}}
{{/TASK_PROPERTIES}}
{{/TASKS}}

### Gateways

{{#GATEWAYS}}
#### {{GATEWAY_NAME}}

- **Type**: {{GATEWAY_TYPE}}
- **Condition**: {{GATEWAY_CONDITION}}
- **Description**: {{GATEWAY_DESCRIPTION}}

{{#GATEWAY_PATHS}}
**Paths**:
{{#PATHS}}
- **{{PATH_NAME}}**: {{PATH_CONDITION}} → {{PATH_DESTINATION}}
{{/PATHS}}
{{/GATEWAY_PATHS}}
{{/GATEWAYS}}

### End Events

{{#END_EVENTS}}
#### {{EVENT_NAME}}

- **Type**: {{EVENT_TYPE}}
- **Result**: {{EVENT_RESULT}}
- **Description**: {{EVENT_DESCRIPTION}}
{{/END_EVENTS}}

## Process Flow

### Happy Path

{{#HAPPY_PATH_STEPS}}
{{STEP_NUMBER}}. **{{STEP_NAME}}**
   - {{STEP_DESCRIPTION}}
   - **Duration**: {{STEP_DURATION}}
   - **Success Criteria**: {{SUCCESS_CRITERIA}}
{{/HAPPY_PATH_STEPS}}

### Alternative Paths

{{#ALTERNATIVE_PATHS}}
#### {{PATH_NAME}}

**Trigger**: {{PATH_TRIGGER}}

**Steps**:
{{#PATH_STEPS}}
{{STEP_NUMBER}}. **{{STEP_NAME}}**: {{STEP_DESCRIPTION}}
{{/PATH_STEPS}}

**Outcome**: {{PATH_OUTCOME}}
{{/ALTERNATIVE_PATHS}}

### Error Handling

{{#ERROR_SCENARIOS}}
#### {{ERROR_NAME}}

- **Trigger**: {{ERROR_TRIGGER}}
- **Impact**: {{ERROR_IMPACT}}
- **Recovery**: {{ERROR_RECOVERY}}
- **Escalation**: {{ERROR_ESCALATION}}

**Recovery Steps**:
{{#RECOVERY_STEPS}}
{{STEP_NUMBER}}. {{RECOVERY_STEP}}
{{/RECOVERY_STEPS}}
{{/ERROR_SCENARIOS}}

## Swimlanes and Responsibilities

{{#SWIMLANES}}
### {{SWIMLANE_NAME}}

**Role**: {{SWIMLANE_ROLE}}  
**Responsibilities**:
{{#RESPONSIBILITIES}}
- {{RESPONSIBILITY}}
{{/RESPONSIBILITIES}}

**Tasks Performed**:
{{#SWIMLANE_TASKS}}
- **{{TASK_NAME}}**: {{TASK_DESCRIPTION}}
{{/SWIMLANE_TASKS}}
{{/SWIMLANES}}

## Data Objects

{{#DATA_OBJECTS}}
### {{DATA_OBJECT_NAME}}

- **Type**: {{DATA_TYPE}}
- **Source**: {{DATA_SOURCE}}
- **Format**: {{DATA_FORMAT}}
- **Lifecycle**: {{DATA_LIFECYCLE}}

**Schema**:
```json
{{DATA_SCHEMA}}
```

**Usage**:
{{#DATA_USAGE}}
- **{{USAGE_CONTEXT}}**: {{USAGE_DESCRIPTION}}
{{/DATA_USAGE}}
{{/DATA_OBJECTS}}

## Performance Metrics

### Key Performance Indicators (KPIs)

{{#KPIS}}
#### {{KPI_NAME}}

- **Definition**: {{KPI_DEFINITION}}
- **Target**: {{KPI_TARGET}}
- **Current**: {{KPI_CURRENT}}
- **Measurement**: {{KPI_MEASUREMENT}}
{{/KPIS}}

### Service Level Agreements (SLAs)

{{#SLAS}}
#### {{SLA_NAME}}

- **Metric**: {{SLA_METRIC}}
- **Target**: {{SLA_TARGET}}
- **Penalty**: {{SLA_PENALTY}}
- **Measurement Period**: {{SLA_PERIOD}}
{{/SLAS}}

## Integration Points

### External Systems

{{#EXTERNAL_INTEGRATIONS}}
#### {{SYSTEM_NAME}}

- **Type**: {{INTEGRATION_TYPE}}
- **Protocol**: {{INTEGRATION_PROTOCOL}}
- **Data Exchange**: {{DATA_EXCHANGE}}
- **Error Handling**: {{INTEGRATION_ERROR_HANDLING}}

**Interface**:
```
{{INTEGRATION_INTERFACE}}
```
{{/EXTERNAL_INTEGRATIONS}}

### Internal Services

{{#INTERNAL_INTEGRATIONS}}
#### {{SERVICE_NAME}}

- **Purpose**: {{SERVICE_PURPOSE}}
- **Method**: {{INTEGRATION_METHOD}}
- **Data Flow**: {{SERVICE_DATA_FLOW}}
- **Dependencies**: {{SERVICE_DEPENDENCIES}}
{{/INTERNAL_INTEGRATIONS}}

## Business Rules

{{#BUSINESS_RULES}}
### Rule {{RULE_NUMBER}}: {{RULE_NAME}}

**Condition**: {{RULE_CONDITION}}

**Action**: {{RULE_ACTION}}

**Exception**: {{RULE_EXCEPTION}}

**Implementation**:
```python
{{RULE_IMPLEMENTATION}}
```
{{/BUSINESS_RULES}}

## Compliance and Governance

### Regulatory Requirements

{{#REGULATORY_REQUIREMENTS}}
- **{{REGULATION_NAME}}**: {{REGULATION_DESCRIPTION}}
  - **Compliance Method**: {{COMPLIANCE_METHOD}}
  - **Evidence**: {{COMPLIANCE_EVIDENCE}}
{{/REGULATORY_REQUIREMENTS}}

### Audit Trail

{{#AUDIT_REQUIREMENTS}}
- **{{AUDIT_POINT}}**: {{AUDIT_DESCRIPTION}}
  - **Data Captured**: {{AUDIT_DATA}}
  - **Retention**: {{AUDIT_RETENTION}}
{{/AUDIT_REQUIREMENTS}}

## Testing and Validation

### Process Testing

{{#PROCESS_TESTS}}
#### Test: {{TEST_NAME}}

**Objective**: {{TEST_OBJECTIVE}}

**Scenario**: {{TEST_SCENARIO}}

**Expected Result**: {{EXPECTED_RESULT}}

**Validation Criteria**: {{VALIDATION_CRITERIA}}
{{/PROCESS_TESTS}}

### User Acceptance Testing

{{#UAT_SCENARIOS}}
#### UAT: {{SCENARIO_NAME}}

**User Role**: {{USER_ROLE}}

**Steps**:
{{#UAT_STEPS}}
{{STEP_NUMBER}}. {{UAT_STEP}}
{{/UAT_STEPS}}

**Acceptance Criteria**: {{UAT_CRITERIA}}
{{/UAT_SCENARIOS}}

## Monitoring and Alerting

### Process Monitoring

{{#MONITORING_POINTS}}
#### {{MONITORING_POINT}}

- **Metric**: {{MONITORING_METRIC}}
- **Threshold**: {{MONITORING_THRESHOLD}}
- **Alert**: {{MONITORING_ALERT}}
- **Action**: {{MONITORING_ACTION}}
{{/MONITORING_POINTS}}

### Dashboard Metrics

{{#DASHBOARD_METRICS}}
- **{{METRIC_NAME}}**: {{METRIC_DESCRIPTION}}
  - **Visualization**: {{METRIC_VISUALIZATION}}
  - **Update Frequency**: {{UPDATE_FREQUENCY}}
{{/DASHBOARD_METRICS}}

## Change Management

### Process Versioning

- **Current Version**: {{CURRENT_VERSION}}
- **Previous Versions**: {{PREVIOUS_VERSIONS}}
- **Change History**: {{CHANGE_HISTORY}}

### Impact Analysis

{{#CHANGE_IMPACTS}}
#### {{CHANGE_TYPE}}

**Impact**: {{CHANGE_IMPACT}}

**Mitigation**: {{CHANGE_MITIGATION}}

**Stakeholders**: {{AFFECTED_STAKEHOLDERS}}
{{/CHANGE_IMPACTS}}

---

**Generated**: {{GENERATION_DATE}}  
**BPMN File**: [{{BPMN_FILE_NAME}}]({{BPMN_FILE_PATH}})  
**Version**: {{VERSION}}  
**Last Updated**: {{LAST_UPDATED}}  
**Process Owner**: {{PROCESS_OWNER}}  
**Validated**: {{VALIDATION_STATUS}}