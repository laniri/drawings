# User Journey - HistoricalInterpretationTracker

**Workflow ID**: `user_journey_historicalinterpretationtracker`
**Type**: User Journey
**Last Updated**: Unknown

## Overview

User journey workflow for HistoricalInterpretationTracker component

## Workflow Elements

| Element ID | Type | Name | Properties |
|------------|------|------|------------|
| start_0 | startEvent | User Starts Journey |  |
| user_task_1 | userTask | Click  => {
                            setSelectedAnalysis(item)
                            setDetailDialogOpen(true)
                           | actor=user |
| user_task_2 | userTask | Click  => setDetailDialogOpen(false) | actor=user |
| user_task_3 | userTask | Click  => {
                onAnalysisSelect(selectedAnalysis.id)
                setDetailDialogOpen(false)
               | actor=user |
| user_task_4 | userTask | Change (e) => setTimeRange(e.target.value as any) | actor=user |
| user_task_5 | userTask | Change (e) => setViewMode(e.target.value as any) | actor=user |
| system_task_6 | serviceTask | Update Loading | actor=system |
| system_task_7 | serviceTask | Update Error | actor=system |
| system_task_8 | serviceTask | Update HistoricalData | actor=system |
| system_task_9 | serviceTask | Update LongitudinalPattern | actor=system |
| system_task_10 | serviceTask | Update DevelopmentalMilestones | actor=system |
| end_11 | endEvent | Journey Complete |  |

## User Actions

- Click  => {
                            setSelectedAnalysis(item)
                            setDetailDialogOpen(true)
                          
- Click  => setDetailDialogOpen(false)
- Click  => {
                onAnalysisSelect(selectedAnalysis.id)
                setDetailDialogOpen(false)
              
- Change (e) => setTimeRange(e.target.value as any)
- Change (e) => setViewMode(e.target.value as any)

## System Responses

- Update Loading
- Update Error
- Update HistoricalData
- Update LongitudinalPattern
- Update DevelopmentalMilestones

## BPMN Diagram

The BPMN 2.0 diagram for this workflow is available as `user_journey_historicalinterpretationtracker.bpmn`.

## Metadata

- **Elements Count**: 12
- **Workflow Type**: user_journey
- **Generated**: Automatically from source code analysis

---

*This documentation was automatically generated from workflow analysis.*
