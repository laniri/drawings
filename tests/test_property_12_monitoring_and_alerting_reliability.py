"""
Property-based test for monitoring and alerting reliability.

**Feature: aws-production-deployment, Property 12: Monitoring and Alerting Reliability**

**Validates: Requirements 5.1, 5.3, 5.5**
"""

import pytest
from hypothesis import given, strategies as st, settings as hypothesis_settings
from unittest.mock import Mock, patch, MagicMock
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging
import threading
from pathlib import Path
import tempfile
import os

from app.services.monitoring_service import MonitoringService, AlertLevel, LogEntry
from app.core.config import settings


class TestMonitoringAndAlertingReliability:
    """Test monitoring and alerting reliability properties."""
    
    def setup_method(self):
        """Set up test environment."""
        # Mock CloudWatch and SNS clients
        self.cloudwatch_patcher = patch('app.services.monitoring_service.boto3.client')
        self.mock_boto3 = self.cloudwatch_patcher.start()
        
        # Create mock clients
        self.mock_cloudwatch = MagicMock()
        self.mock_sns = MagicMock()
        
        def mock_client(service_name, **kwargs):
            if service_name == 'cloudwatch':
                return self.mock_cloudwatch
            elif service_name == 'sns':
                return self.mock_sns
            else:
                return MagicMock()
        
        self.mock_boto3.side_effect = mock_client
        
        # Mock successful AWS responses
        self.mock_cloudwatch.put_metric_data.return_value = {
            'ResponseMetadata': {'HTTPStatusCode': 200}
        }
        self.mock_sns.publish.return_value = {
            'ResponseMetadata': {'HTTPStatusCode': 200},
            'MessageId': 'test-message-id'
        }
        
        # Create temporary log directory
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "test_monitoring.log"
        
    def teardown_method(self):
        """Clean up test environment."""
        self.cloudwatch_patcher.stop()
        
        # Clean up temporary files
        if self.temp_dir and Path(self.temp_dir).exists():
            import shutil
            shutil.rmtree(self.temp_dir)
    
    @given(
        error_count=st.integers(min_value=1, max_value=100),
        error_types=st.lists(
            st.sampled_from(['DatabaseError', 'ModelError', 'StorageError', 'ConfigurationError', 'SecurityError', 'NetworkError', 'ValidationError']),
            min_size=1,
            max_size=10
        ),
        correlation_ids=st.lists(
            st.text(min_size=8, max_size=36, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
            min_size=1,
            max_size=20
        )
    )
    @hypothesis_settings(max_examples=30, deadline=None)
    def test_error_logging_with_correlation_ids_reliability(
        self,
        error_count: int,
        error_types: List[str],
        correlation_ids: List[str]
    ):
        """
        **Feature: aws-production-deployment, Property 12: Monitoring and Alerting Reliability**
        
        **Validates: Requirements 5.1, 5.3, 5.5**
        
        For any system error or performance issue, appropriate alerts should be generated 
        and logs should be collected with proper correlation IDs.
        """
        # Create monitoring service with test configuration
        with patch.dict(os.environ, {
            'ENVIRONMENT': 'production',
            'AWS_REGION': 'eu-west-1',
            'SNS_ALERT_TOPIC_ARN': 'arn:aws:sns:eu-west-1:123456789012:test-alerts'
        }), patch('app.services.monitoring_service.settings') as mock_settings:
            # Mock settings to indicate production environment
            mock_settings.is_production = True
            mock_settings.aws_region = 'eu-west-1'
            
            monitoring_service = MonitoringService(
                log_file_path=str(self.log_file),
                cloudwatch_namespace="Test/Monitoring"
            )
        
        # Generate errors with correlation IDs
        logged_entries = []
        sent_alerts = []
        
        for i in range(min(error_count, len(error_types), len(correlation_ids))):
            error_type = error_types[i % len(error_types)]
            correlation_id = correlation_ids[i % len(correlation_ids)]
            
            # Log error with correlation ID
            log_entry = monitoring_service.log_error(
                message=f"Test error {i}: {error_type}",
                error_type=error_type,
                correlation_id=correlation_id,
                details={"error_index": i, "test_run": True}
            )
            
            logged_entries.append(log_entry)
            
            # Check if alert should be sent (based on error severity)
            if monitoring_service.should_send_alert(error_type, correlation_id):
                alert_result = monitoring_service.send_alert(
                    level=AlertLevel.ERROR,
                    message=f"Alert for {error_type}",
                    correlation_id=correlation_id,
                    details={"error_type": error_type, "count": i + 1}
                )
                sent_alerts.append(alert_result)
        
        # Property 1: All errors must be logged with correlation IDs
        assert len(logged_entries) > 0, "No log entries were created"
        
        for entry in logged_entries:
            assert entry.correlation_id is not None, f"Log entry missing correlation ID: {entry}"
            assert entry.correlation_id.strip() != "", f"Empty correlation ID in entry: {entry}"
            assert entry.level in ['ERROR', 'CRITICAL'], f"Invalid log level: {entry.level}"
            assert entry.message is not None and entry.message.strip() != "", f"Empty log message: {entry}"
        
        # Property 2: Correlation IDs must be preserved across logging and alerting
        correlation_ids_in_logs = {entry.correlation_id for entry in logged_entries}
        correlation_ids_in_alerts = {alert.correlation_id for alert in sent_alerts if alert.correlation_id}
        
        # All alert correlation IDs should be present in logs
        for alert_correlation_id in correlation_ids_in_alerts:
            assert alert_correlation_id in correlation_ids_in_logs, \
                f"Alert correlation ID {alert_correlation_id} not found in logs"
        
        # Property 3: CloudWatch metrics must be sent for all errors
        cloudwatch_calls = self.mock_cloudwatch.put_metric_data.call_args_list
        assert len(cloudwatch_calls) > 0, "No CloudWatch metrics were sent"
        
        # Verify metric data structure
        for call in cloudwatch_calls:
            args, kwargs = call
            assert 'Namespace' in kwargs, "CloudWatch call missing namespace"
            assert 'MetricData' in kwargs, "CloudWatch call missing metric data"
            
            metric_data = kwargs['MetricData']
            assert isinstance(metric_data, list), "MetricData should be a list"
            assert len(metric_data) > 0, "MetricData should not be empty"
            
            for metric in metric_data:
                assert 'MetricName' in metric, f"Metric missing name: {metric}"
                assert 'Value' in metric, f"Metric missing value: {metric}"
                assert 'Unit' in metric, f"Metric missing unit: {metric}"
                assert 'Timestamp' in metric, f"Metric missing timestamp: {metric}"
        
        # Property 4: Log file must contain structured entries with correlation IDs
        if self.log_file.exists():
            log_content = self.log_file.read_text()
            
            # Check that correlation IDs appear in log file
            for correlation_id in correlation_ids_in_logs:
                assert correlation_id in log_content, \
                    f"Correlation ID {correlation_id} not found in log file"
        
        # Property 5: Alerts must be sent for critical errors
        if sent_alerts:
            sns_calls = self.mock_sns.publish.call_args_list
            assert len(sns_calls) > 0, "No SNS alerts were sent despite having alerts to send"
            
            for call in sns_calls:
                args, kwargs = call
                assert 'TopicArn' in kwargs, "SNS call missing TopicArn"
                assert 'Message' in kwargs, "SNS call missing Message"
                assert 'Subject' in kwargs, "SNS call missing Subject"
                
                # Verify message structure
                message = kwargs['Message']
                if isinstance(message, str):
                    try:
                        message_data = json.loads(message)
                        assert 'correlation_id' in message_data, "Alert message missing correlation_id"
                        assert 'timestamp' in message_data, "Alert message missing timestamp"
                        assert 'level' in message_data, "Alert message missing level"
                    except json.JSONDecodeError:
                        # If not JSON, ensure it contains correlation ID as text
                        correlation_id_found = any(
                            corr_id in message for corr_id in correlation_ids_in_logs
                        )
                        assert correlation_id_found, f"No correlation ID found in alert message: {message}"
    
    @given(
        performance_issues=st.lists(
            st.dictionaries(
                keys=st.sampled_from(['cpu_usage', 'memory_usage', 'response_time', 'error_rate']),
                values=st.floats(min_value=0.0, max_value=100.0),
                min_size=1,
                max_size=4
            ),
            min_size=1,
            max_size=20
        ),
        thresholds=st.dictionaries(
            keys=st.sampled_from(['cpu_usage', 'memory_usage', 'response_time', 'error_rate']),
            values=st.floats(min_value=50.0, max_value=95.0),
            min_size=1,
            max_size=4
        )
    )
    @hypothesis_settings(max_examples=20, deadline=None)
    def test_performance_monitoring_and_alerting_reliability(
        self,
        performance_issues: List[Dict[str, float]],
        thresholds: Dict[str, float]
    ):
        """
        **Feature: aws-production-deployment, Property 12: Monitoring and Alerting Reliability**
        
        **Validates: Requirements 5.1, 5.3, 5.5**
        
        For any performance issue that exceeds configured thresholds, appropriate 
        monitoring metrics and alerts should be generated reliably.
        """
        # Create monitoring service with performance thresholds
        with patch.dict(os.environ, {
            'ENVIRONMENT': 'production',
            'AWS_REGION': 'eu-west-1',
            'SNS_ALERT_TOPIC_ARN': 'arn:aws:sns:eu-west-1:123456789012:test-alerts'
        }), patch('app.services.monitoring_service.settings') as mock_settings:
            # Mock settings to indicate production environment
            mock_settings.is_production = True
            mock_settings.aws_region = 'eu-west-1'
            
            monitoring_service = MonitoringService(
                log_file_path=str(self.log_file),
                cloudwatch_namespace="Test/Performance",
                performance_thresholds=thresholds
            )
        
        # Record performance metrics
        recorded_metrics = []
        triggered_alerts = []
        
        for i, performance_data in enumerate(performance_issues):
            correlation_id = f"perf-test-{i}-{int(time.time())}"
            
            # Record performance metrics
            metric_result = monitoring_service.record_performance_metrics(
                metrics=performance_data,
                correlation_id=correlation_id
            )
            recorded_metrics.append(metric_result)
            
            # Check for threshold violations and alerts
            for metric_name, value in performance_data.items():
                if metric_name in thresholds and value > thresholds[metric_name]:
                    alert = monitoring_service.send_performance_alert(
                        metric_name=metric_name,
                        current_value=value,
                        threshold=thresholds[metric_name],
                        correlation_id=correlation_id
                    )
                    triggered_alerts.append(alert)
        
        # Property 1: All performance metrics must be recorded
        assert len(recorded_metrics) == len(performance_issues), \
            f"Expected {len(performance_issues)} recorded metrics, got {len(recorded_metrics)}"
        
        for metric_result in recorded_metrics:
            assert metric_result.success, f"Failed to record metric: {metric_result.error_message}"
            assert metric_result.correlation_id is not None, "Metric result missing correlation ID"
        
        # Property 2: CloudWatch must receive all performance metrics
        cloudwatch_calls = self.mock_cloudwatch.put_metric_data.call_args_list
        assert len(cloudwatch_calls) > 0, "No performance metrics sent to CloudWatch"
        
        # Count total metrics sent
        total_metrics_sent = 0
        for call in cloudwatch_calls:
            args, kwargs = call
            metric_data = kwargs.get('MetricData', [])
            total_metrics_sent += len(metric_data)
        
        # Should have at least one metric per performance issue
        assert total_metrics_sent >= len(performance_issues), \
            f"Expected at least {len(performance_issues)} metrics, sent {total_metrics_sent}"
        
        # Property 3: Threshold violations must trigger alerts
        expected_violations = 0
        for performance_data in performance_issues:
            for metric_name, value in performance_data.items():
                if metric_name in thresholds and value > thresholds[metric_name]:
                    expected_violations += 1
        
        if expected_violations > 0:
            assert len(triggered_alerts) > 0, \
                f"Expected alerts for {expected_violations} threshold violations, got {len(triggered_alerts)}"
            
            # Verify alert structure
            for alert in triggered_alerts:
                assert alert.success, f"Failed to send alert: {alert.error_message}"
                assert alert.correlation_id is not None, "Alert missing correlation ID"
                assert alert.alert_id is not None, "Alert missing alert ID"
        
        # Property 4: Alert correlation IDs must match metric correlation IDs
        if triggered_alerts:
            metric_correlation_ids = {result.correlation_id for result in recorded_metrics}
            alert_correlation_ids = {alert.correlation_id for alert in triggered_alerts}
            
            for alert_correlation_id in alert_correlation_ids:
                assert alert_correlation_id in metric_correlation_ids, \
                    f"Alert correlation ID {alert_correlation_id} not found in metrics"
        
        # Property 5: SNS alerts must be properly formatted
        if triggered_alerts:
            sns_calls = self.mock_sns.publish.call_args_list
            assert len(sns_calls) > 0, "No SNS alerts sent despite triggered alerts"
            
            for call in sns_calls:
                args, kwargs = call
                message = kwargs.get('Message', '')
                
                # Verify alert contains performance information
                assert any(
                    metric_name in message.lower() 
                    for metric_name in ['cpu', 'memory', 'response', 'error']
                ), f"Performance alert missing metric information: {message}"
    
    @given(
        log_entries_count=st.integers(min_value=10, max_value=100),
        log_levels=st.lists(
            st.sampled_from(['INFO', 'WARNING', 'ERROR', 'CRITICAL']),
            min_size=1,
            max_size=4
        )
    )
    @hypothesis_settings(max_examples=15, deadline=None)
    def test_structured_logging_reliability(
        self,
        log_entries_count: int,
        log_levels: List[str]
    ):
        """
        **Feature: aws-production-deployment, Property 12: Monitoring and Alerting Reliability**
        
        **Validates: Requirements 5.1, 5.3, 5.5**
        
        For any logging operation, structured logs with correlation IDs must be 
        reliably written and properly formatted.
        """
        # Create monitoring service
        with patch.dict(os.environ, {
            'ENVIRONMENT': 'production',
            'AWS_REGION': 'eu-west-1'
        }), patch('app.services.monitoring_service.settings') as mock_settings:
            # Mock settings to indicate production environment
            mock_settings.is_production = True
            mock_settings.aws_region = 'eu-west-1'
            
            monitoring_service = MonitoringService(
                log_file_path=str(self.log_file),
                cloudwatch_namespace="Test/Logging"
            )
        
        # Generate structured log entries
        generated_entries = []
        
        for i in range(log_entries_count):
            level = log_levels[i % len(log_levels)]
            correlation_id = f"log-test-{i}-{int(time.time() * 1000)}"
            
            log_entry = monitoring_service.log_structured(
                level=level,
                message=f"Test log entry {i}",
                correlation_id=correlation_id,
                component="test_component",
                operation="test_operation",
                details={
                    "entry_index": i,
                    "test_data": f"data_{i}",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            generated_entries.append(log_entry)
        
        # Property 1: All log entries must be successfully created
        assert len(generated_entries) == log_entries_count, \
            f"Expected {log_entries_count} log entries, got {len(generated_entries)}"
        
        for entry in generated_entries:
            assert entry.success, f"Failed to create log entry: {entry.error_message}"
            assert entry.correlation_id is not None, "Log entry missing correlation ID"
            assert entry.timestamp is not None, "Log entry missing timestamp"
        
        # Property 2: Log file must contain all entries with proper structure
        if self.log_file.exists():
            log_content = self.log_file.read_text()
            
            for entry in generated_entries:
                # Check correlation ID is in log file
                assert entry.correlation_id in log_content, \
                    f"Correlation ID {entry.correlation_id} not found in log file"
                
                # Check structured format (JSON or key-value pairs)
                entry_found = False
                for line in log_content.split('\n'):
                    if entry.correlation_id in line:
                        entry_found = True
                        # Verify structured format
                        assert any(
                            marker in line for marker in ['correlation_id', 'component', 'operation']
                        ), f"Log line missing structured format: {line}"
                        break
                
                assert entry_found, f"Log entry not found in file: {entry.correlation_id}"
        
        # Property 3: CloudWatch logs must receive structured entries
        # Note: In real implementation, this would check CloudWatch Logs API calls
        # For this test, we verify the monitoring service tracks the entries
        service_stats = monitoring_service.get_service_stats()
        assert service_stats['total_log_entries'] >= log_entries_count, \
            f"Service should track at least {log_entries_count} log entries"
        
        # Property 4: Correlation IDs must be unique within the test run
        correlation_ids = [entry.correlation_id for entry in generated_entries]
        unique_correlation_ids = set(correlation_ids)
        
        assert len(unique_correlation_ids) == len(correlation_ids), \
            f"Duplicate correlation IDs found: {len(correlation_ids)} total, {len(unique_correlation_ids)} unique"
        
        # Property 5: Log levels must be properly categorized
        level_counts = {}
        for entry in generated_entries:
            level = entry.level
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Verify all requested levels were used
        for level in log_levels:
            assert level in [entry.level for entry in generated_entries], \
                f"Log level {level} was not used in any entries"