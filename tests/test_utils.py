"""Unit tests for utility modules (logger, cost_tracker)."""

import pytest
from src.utils.logger import logger
from src.utils.cost_tracker import CostTracker


class TestLogger:
    """Test suite for logger module."""

    def test_logger_import(self):
        """Test that logger can be imported."""
        assert logger is not None

    def test_logger_has_basic_methods(self):
        """Test that logger has expected methods."""
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'warning')
        assert hasattr(logger, 'debug')

    def test_logger_info(self):
        """Test that logger.info works without errors."""
        try:
            logger.info("Test info message")
        except Exception as e:
            pytest.fail(f"logger.info raised exception: {e}")

    def test_logger_error(self):
        """Test that logger.error works without errors."""
        try:
            logger.error("Test error message")
        except Exception as e:
            pytest.fail(f"logger.error raised exception: {e}")

    def test_logger_with_extra_context(self):
        """Test that logger can handle extra context."""
        try:
            logger.info("Test with context", extra={"key": "value"})
        except Exception as e:
            pytest.fail(f"logger with extra context raised exception: {e}")


class TestCostTracker:
    """Test suite for CostTracker class."""

    def test_cost_tracker_init_default(self):
        """Test CostTracker initialization with defaults."""
        tracker = CostTracker()
        assert tracker.daily_budget == 1.0
        assert tracker.current_spend == 0.0

    def test_cost_tracker_init_custom_budget(self):
        """Test CostTracker initialization with custom budget."""
        tracker = CostTracker(daily_budget=5.0)
        assert tracker.daily_budget == 5.0
        assert tracker.current_spend == 0.0

    def test_cost_tracker_has_record_method(self):
        """Test that CostTracker has record_api_call method."""
        tracker = CostTracker()
        assert hasattr(tracker, 'record_api_call')
        assert callable(tracker.record_api_call)

    def test_record_api_call_signature(self):
        """Test that record_api_call accepts correct parameters."""
        tracker = CostTracker()
        try:
            tracker.record_api_call(
                provider="xai",
                model="grok-4-fast-reasoning",
                input_tokens=1000,
                output_tokens=500,
                cost=0.0007
            )
        except TypeError as e:
            pytest.fail(f"record_api_call signature error: {e}")

    def test_cost_tracker_accepts_zero_budget(self):
        """Test CostTracker with zero budget (unlimited mode)."""
        tracker = CostTracker(daily_budget=0.0)
        assert tracker.daily_budget == 0.0

    def test_cost_tracker_accepts_large_budget(self):
        """Test CostTracker with large budget."""
        tracker = CostTracker(daily_budget=1000.0)
        assert tracker.daily_budget == 1000.0

    def test_cost_tracker_current_spend_is_numeric(self):
        """Test that current_spend is numeric type."""
        tracker = CostTracker()
        assert isinstance(tracker.current_spend, (int, float))

    def test_cost_tracker_daily_budget_is_numeric(self):
        """Test that daily_budget is numeric type."""
        tracker = CostTracker(daily_budget=2.5)
        assert isinstance(tracker.daily_budget, (int, float))


class TestCostTrackerIntegration:
    """Integration tests for cost tracking scenarios."""

    def test_multiple_api_calls(self):
        """Test recording multiple API calls."""
        tracker = CostTracker(daily_budget=10.0)

        # Record several calls
        for i in range(5):
            tracker.record_api_call(
                provider="xai",
                model="grok-4",
                input_tokens=1000 * i,
                output_tokens=500 * i,
                cost=0.001 * i
            )

    def test_different_providers(self):
        """Test recording calls from different providers."""
        tracker = CostTracker()

        providers = [
            ("xai", "grok-4-fast-reasoning"),
            ("together", "glm-4-9b-chat"),
            ("openai", "gpt-4"),
        ]

        for provider, model in providers:
            tracker.record_api_call(
                provider=provider,
                model=model,
                input_tokens=1000,
                output_tokens=500,
                cost=0.001
            )

    def test_zero_cost_calls(self):
        """Test recording calls with zero cost."""
        tracker = CostTracker()
        tracker.record_api_call(
            provider="local",
            model="test-model",
            input_tokens=0,
            output_tokens=0,
            cost=0.0
        )

    def test_high_volume_calls(self):
        """Test recording high volume of API calls."""
        tracker = CostTracker(daily_budget=100.0)

        # Simulate high volume
        for i in range(100):
            tracker.record_api_call(
                provider="xai",
                model="grok-4",
                input_tokens=100,
                output_tokens=50,
                cost=0.00001
            )
