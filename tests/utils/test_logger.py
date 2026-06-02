import logging
import os
import sys
from pathlib import Path

from src.utils.logger import ProjectLogger


def test_logger_creates_file(tmp_path: Path) -> None:
    log_file = tmp_path / "test_logs.txt"

    # Reset singleton array for testing purposes
    ProjectLogger._ProjectLogger__instance = None

    # Initialize logger
    ProjectLogger.initialize(log_file=str(log_file))

    # Write some logs natively
    logger = logging.getLogger("test")
    logger.info("This is a test log")

    # Capture standard print
    print("This is a random print")

    # Assert logs exist
    assert os.path.exists(log_file)
    with open(log_file) as f:
        content = f.read()
        assert "This is a test log" in content
        assert "This is a random print" in content

    # Teardown
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    ProjectLogger._ProjectLogger__instance = None
