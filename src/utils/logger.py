import logging
import sys
from typing import Optional


class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger: logging.Logger, log_level: int = logging.INFO) -> None:
        """
        Initializes the stream redirector.
        """
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf: str) -> None:
        """
        Redirects standard string writes to the logger.
        """
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self) -> None:
        """
        Flush method for compatibility with file-like objects.
        """
        pass


class ProjectLogger:
    """
    Centralized logging configuration for the project.
    Ensures that all generic output and exceptions are written to a file.
    """

    __instance: Optional["ProjectLogger"] = None

    def __init__(self, log_file: str = "logs.txt", level: int = logging.INFO) -> None:
        """
        Configures the root logger and redirects standard streams.
        """
        if ProjectLogger.__instance is not None:
            return

        self._configure_logging(log_file, level)
        self._redirect_std_streams()

        ProjectLogger.__instance = self

    def _configure_logging(self, log_file: str, level: int) -> None:
        """
        Sets up the root logger with a file handler and a console handler.
        """
        self.logger = logging.getLogger()
        self.logger.setLevel(level)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.__stdout__)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def _redirect_std_streams(self) -> None:
        """
        Redirects stdout and stderr to the configured logger.
        """
        sys.stdout = StreamToLogger(self.logger, logging.INFO)  # type: ignore
        sys.stderr = StreamToLogger(self.logger, logging.ERROR)  # type: ignore

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Returns a logger instance with the specified name.
        """
        return logging.getLogger(name)

    @classmethod
    def initialize(cls, log_file: str = "logs.txt") -> None:
        """
        Initializes the singleton logger system.
        """
        if cls.__instance is None:
            cls(log_file=log_file)
