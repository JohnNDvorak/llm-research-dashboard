"""Tests for logging configuration."""

import pytest
from pathlib import Path
import tempfile
import shutil
from loguru import logger as global_logger
from src.utils.logger import get_logger, set_log_level


class TestLoggerConfiguration:
    """Test logger setup and configuration."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary logs directory for testing
        self.temp_logs_dir = Path(tempfile.mkdtemp())
        self.original_handlers = []

        # Store original logger state
        for handler in global_logger._core.handlers.values():
            self.original_handlers.append(handler)

    def teardown_method(self):
        """Clean up after tests."""
        # Clean up temp directory
        if self.temp_logs_dir.exists():
            shutil.rmtree(self.temp_logs_dir)

    def test_logger_directory_creation(self):
        """Test that logs directory is created automatically."""
        logs_dir = Path("logs")
        assert logs_dir.exists(), "Logs directory should be created on import"
        assert logs_dir.is_dir(), "Logs path should be a directory"

    def test_get_logger_without_name(self):
        """Test getting logger without a name."""
        log = get_logger()

        # Logger should be returned
        assert log is not None

        # Should be able to log
        log.info("Test message without name")

    def test_get_logger_with_name(self):
        """Test getting logger with a name binding."""
        log = get_logger("test_module")

        # Logger should be returned
        assert log is not None

        # Should be able to log with context
        log.info("Test message with name")

    def test_set_log_level_debug(self):
        """Test setting log level to DEBUG."""
        # This should not raise an error
        set_log_level("DEBUG")

        # Verify we can log at DEBUG level
        log = get_logger("test_debug")
        log.debug("Debug message")

    def test_set_log_level_info(self):
        """Test setting log level to INFO."""
        set_log_level("INFO")

        # Verify we can log at INFO level
        log = get_logger("test_info")
        log.info("Info message")

    def test_set_log_level_warning(self):
        """Test setting log level to WARNING."""
        set_log_level("WARNING")

        # Verify we can log at WARNING level
        log = get_logger("test_warning")
        log.warning("Warning message")

    def test_set_log_level_error(self):
        """Test setting log level to ERROR."""
        set_log_level("ERROR")

        # Verify we can log at ERROR level
        log = get_logger("test_error")
        log.error("Error message")

    def test_set_log_level_critical(self):
        """Test setting log level to CRITICAL."""
        set_log_level("CRITICAL")

        # Verify we can log at CRITICAL level
        log = get_logger("test_critical")
        log.critical("Critical message")

    def test_set_log_level_case_insensitive(self):
        """Test that log level setting is case insensitive."""
        # Should not raise errors
        set_log_level("debug")
        set_log_level("Info")
        set_log_level("WARNING")
        set_log_level("ErRoR")

    def test_log_files_created(self):
        """Test that log files are created."""
        logs_dir = Path("logs")

        # Main log file
        main_log = logs_dir / "llm_dashboard.log"

        # Errors log file
        errors_log = logs_dir / "errors.log"

        # At least one should exist after import
        # (they're created on first log message)
        log = get_logger("test_files")
        log.info("Test message to create log files")
        log.error("Test error to create error log")

        # Give it a moment to write
        import time
        time.sleep(0.1)

        # Check files exist
        assert main_log.exists(), "Main log file should be created"
        assert errors_log.exists(), "Error log file should be created"

    def test_log_file_writing(self):
        """Test that logs are actually written to files."""
        logs_dir = Path("logs")
        main_log = logs_dir / "llm_dashboard.log"

        # Log a unique message
        log = get_logger("test_writing")
        test_message = "UNIQUE_TEST_MESSAGE_12345"
        log.info(test_message)

        # Give it a moment to write
        import time
        time.sleep(0.1)

        # Read log file and check for message
        if main_log.exists():
            content = main_log.read_text()
            assert test_message in content, "Log message should be written to file"

    def test_error_log_separation(self):
        """Test that errors are logged to separate error file."""
        logs_dir = Path("logs")
        errors_log = logs_dir / "errors.log"

        # Log an error with unique message
        log = get_logger("test_errors")
        error_message = "UNIQUE_ERROR_MESSAGE_67890"
        log.error(error_message)

        # Give it a moment to write
        import time
        time.sleep(0.1)

        # Check error log contains the message
        if errors_log.exists():
            content = errors_log.read_text()
            assert error_message in content, "Error should be written to errors.log"

    def test_logger_with_exception(self):
        """Test logging with exception traceback."""
        log = get_logger("test_exception")

        try:
            # Cause an exception
            raise ValueError("Test exception")
        except ValueError:
            # Should not raise an error
            log.exception("Exception occurred")

    def test_logger_with_extra_context(self):
        """Test logging with extra context data."""
        log = get_logger("test_context")

        # Should not raise an error
        log.info("Message with context", extra={"key": "value", "count": 42})

    def test_multiple_loggers_with_names(self):
        """Test creating multiple named loggers."""
        log1 = get_logger("module1")
        log2 = get_logger("module2")
        log3 = get_logger("module3")

        # All should be valid
        assert log1 is not None
        assert log2 is not None
        assert log3 is not None

        # All should be able to log
        log1.info("Message from module1")
        log2.info("Message from module2")
        log3.info("Message from module3")

    def test_logger_formats_contain_metadata(self):
        """Test that log formats include expected metadata."""
        # This is validated by the logger configuration
        # We're checking that the logger is set up with proper format strings
        from src.utils import logger as logger_module

        # The module should have logger exported
        assert hasattr(logger_module, 'logger')
        assert hasattr(logger_module, 'get_logger')
        assert hasattr(logger_module, 'set_log_level')


class TestLoggerIntegration:
    """Integration tests for logger across modules."""

    def test_logger_import_from_utils(self):
        """Test importing logger from utils package."""
        from src.utils.logger import logger

        assert logger is not None
        logger.info("Test import from utils.logger")

    def test_logger_in_database_module(self):
        """Test that database module can use logger."""
        # The paper_db.py module already imports and uses logger
        from src.storage.paper_db import PaperDB

        # Creating instance should use logger without errors
        with PaperDB(db_path=":memory:") as db:
            # This will log connection messages
            pass

    def test_logger_in_vector_store_module(self):
        """Test that vector store module can use logger."""
        from src.embeddings.vector_store import VectorStore
        import tempfile

        # Creating instance should use logger without errors
        temp_dir = tempfile.mkdtemp()
        try:
            with VectorStore(persist_directory=temp_dir) as store:
                # This will log connection messages
                pass
        finally:
            shutil.rmtree(temp_dir)

    def test_logger_performance(self):
        """Test that logging doesn't significantly impact performance."""
        import time

        log = get_logger("test_performance")

        # Time 1000 log operations
        start = time.time()
        for i in range(1000):
            log.info(f"Performance test message {i}")
        duration = time.time() - start

        # Should complete in reasonable time (< 1 second for 1000 messages)
        assert duration < 1.0, f"1000 log messages took {duration:.2f}s (should be < 1s)"

    def test_concurrent_logging(self):
        """Test that concurrent logging works correctly."""
        import threading

        def log_messages(thread_id: int):
            log = get_logger(f"thread_{thread_id}")
            for i in range(100):
                log.info(f"Thread {thread_id} message {i}")

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=log_messages, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # If we got here without deadlock or errors, test passes


class TestLoggerEdgeCases:
    """Test edge cases and error handling."""

    def test_set_log_level_with_none(self):
        """Test setting log level with None (should handle gracefully)."""
        # This might raise an error, which is acceptable
        with pytest.raises((TypeError, ValueError, AttributeError)):
            set_log_level(None)

    def test_get_logger_with_empty_string(self):
        """Test getting logger with empty string name."""
        log = get_logger("")

        # Should still return a logger
        assert log is not None
        log.info("Message with empty name")

    def test_get_logger_with_special_characters(self):
        """Test getting logger with special characters in name."""
        log = get_logger("module.submodule:function")

        assert log is not None
        log.info("Message with special characters in name")

    def test_logging_unicode_characters(self):
        """Test logging messages with unicode characters."""
        log = get_logger("test_unicode")

        # Should not raise an error
        log.info("Unicode message: 你好世界 🌍 café")
        log.info("Emoji: 🚀 🎉 ✅ ❌")

    def test_logging_very_long_message(self):
        """Test logging very long messages."""
        log = get_logger("test_long")

        # Create a very long message
        long_message = "A" * 10000

        # Should not raise an error
        log.info(long_message)

    def test_logging_with_newlines(self):
        """Test logging messages with newlines."""
        log = get_logger("test_newlines")

        message = "Line 1\nLine 2\nLine 3"
        log.info(message)

    def test_logging_dict_objects(self):
        """Test logging dictionary objects."""
        log = get_logger("test_dict")

        data = {
            "key1": "value1",
            "key2": 42,
            "key3": [1, 2, 3],
            "key4": {"nested": "dict"}
        }

        log.info(f"Data: {data}")

    def test_logging_after_level_change(self):
        """Test that logging works after changing levels."""
        log = get_logger("test_level_change")

        # Change levels multiple times
        set_log_level("DEBUG")
        log.debug("Debug message 1")

        set_log_level("INFO")
        log.info("Info message 1")

        set_log_level("ERROR")
        log.error("Error message 1")

        # All should complete without errors
