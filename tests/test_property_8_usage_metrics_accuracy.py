"""
Property-based test for usage metrics accuracy.

**Property 8: Usage Metrics Accuracy**
*For any* application operation that generates metrics, the recorded metrics should accurately reflect the actual system behavior within acceptable measurement tolerances
**Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5**
"""

import pytest
from datetime import datetime, timedelta, timezone
from hypothesis import given, strategies as st, settings
from app.services.usage_metrics_service import UsageMetricsService


class TestUsageMetricsAccuracy:
    """Test suite for usage metrics accuracy property."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create a fresh service instance for each test
        self.metrics_service = UsageMetricsService()
        # Clear any existing data
        with self.metrics_service._lock:
            self.metrics_service._analysis_metrics.clear()
            self.metrics_service._session_metrics.clear()
            self.metrics_service._system_health_metrics.clear()
            self.metrics_service._daily_stats.clear()
    
    @given(
        session_count=st.integers(min_value=1, max_value=10),
        analyses_per_session=st.integers(min_value=0, max_value=5)
    )
    @settings(max_examples=10, deadline=5000)
    def test_session_metrics_accuracy(self, session_count, analyses_per_session):
        """
        Test that session metrics accurately reflect actual session activity.
        
        **Property 8: Usage Metrics Accuracy**
        **Validates: Requirements 10.4, 10.5**
        """
        # Create sessions and track expected metrics
        session_ids = []
        
        for i in range(session_count):
            session_id = f"test_session_{i}"
            ip_address = f"192.168.1.{i % 255}"
            user_agent = f"TestAgent/{i}"
            
            # Start session
            self.metrics_service.start_session(
                session_id=session_id,
                ip_address=ip_address,
                user_agent=user_agent
            )
            session_ids.append(session_id)
            
            # Simulate session activity
            for _ in range(analyses_per_session):
                self.metrics_service.update_session_activity(session_id)
        
        # Verify session count accuracy
        with self.metrics_service._lock:
            active_sessions = len(self.metrics_service._session_metrics)
            assert active_sessions == session_count, \
                f"Expected {session_count} active sessions, got {active_sessions}"
        
        # Clean up sessions
        for session_id in session_ids:
            self.metrics_service.end_session(session_id)
    
    @given(
        analysis_count=st.integers(min_value=1, max_value=20),
        anomaly_rate=st.floats(min_value=0.0, max_value=1.0),
        processing_time_base=st.floats(min_value=0.1, max_value=2.0)
    )
    @settings(max_examples=5, deadline=10000)  # Reduced examples for faster testing
    def test_analysis_metrics_accuracy(self, analysis_count, anomaly_rate, processing_time_base):
        """
        Test that analysis metrics accurately reflect actual analysis operations.
        
        **Property 8: Usage Metrics Accuracy**
        **Validates: Requirements 10.1, 10.2, 10.3**
        """
        # Clear metrics at start of each test
        with self.metrics_service._lock:
            self.metrics_service._analysis_metrics.clear()
        
        expected_anomaly_count = 0
        expected_normal_count = 0
        
        # Simulate analyses
        for i in range(analysis_count):
            # Determine if this analysis should be anomalous
            is_anomaly = (i / analysis_count) < anomaly_rate
            
            # Generate processing time
            processing_time = processing_time_base + (i * 0.01)
            
            # Record analysis
            self.metrics_service.record_analysis(
                processing_time=processing_time,
                age_group=f"age_{5 + (i % 5)}",
                anomaly_detected=is_anomaly,
                error_occurred=False
            )
            
            if is_anomaly:
                expected_anomaly_count += 1
            else:
                expected_normal_count += 1
        
        # Verify analysis metrics accuracy
        with self.metrics_service._lock:
            total_analyses = len(self.metrics_service._analysis_metrics)
            assert total_analyses == analysis_count, \
                f"Expected {analysis_count} total analyses, got {total_analyses}"
            
            # Count anomalies and normal results
            actual_anomaly_count = sum(1 for m in self.metrics_service._analysis_metrics if m.anomaly_detected)
            actual_normal_count = sum(1 for m in self.metrics_service._analysis_metrics if not m.anomaly_detected)
            
            assert actual_anomaly_count == expected_anomaly_count, \
                f"Expected {expected_anomaly_count} anomalies, got {actual_anomaly_count}"
            
            assert actual_normal_count == expected_normal_count, \
                f"Expected {expected_normal_count} normal results, got {actual_normal_count}"
    
    def test_dashboard_stats_structure(self):
        """
        Test that dashboard stats return the expected structure and data types.
        
        **Property 8: Usage Metrics Accuracy**
        **Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5**
        """
        # Add some test data
        self.metrics_service.record_analysis(
            processing_time=0.5,
            age_group="age_5",
            anomaly_detected=True,
            error_occurred=False
        )
        
        self.metrics_service.record_analysis(
            processing_time=0.3,
            age_group="age_6",
            anomaly_detected=False,
            error_occurred=False
        )
        
        # Get dashboard stats
        stats = self.metrics_service.get_dashboard_stats()
        
        # Verify structure and accuracy
        assert isinstance(stats, dict), "Dashboard stats should be a dictionary"
        
        # Check required fields
        required_fields = [
            'total_analyses', 'daily_analyses', 'weekly_analyses', 'monthly_analyses',
            'average_processing_time', 'active_sessions', 'error_rate', 'uptime_percentage'
        ]
        
        for field in required_fields:
            assert field in stats, f"Missing required field: {field}"
        
        # Verify data accuracy
        assert stats['total_analyses'] == 2, f"Expected 2 analyses, got {stats['total_analyses']}"
        assert stats['daily_analyses'] == 2, f"Expected 2 daily analyses, got {stats['daily_analyses']}"
        assert stats['average_processing_time'] == 0.4, f"Expected 0.4 avg time, got {stats['average_processing_time']}"
        assert stats['error_rate'] == 0.0, f"Expected 0.0 error rate, got {stats['error_rate']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])