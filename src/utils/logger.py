"""Centralized logging configuration for the project."""

import logging
import sys
import types


class StreamToLogger:
    """Fake file-like stream object that redirects writes to a logger instance."""

    def __init__(self, logger: logging.Logger, log_level: int = logging.INFO) -> None:
        """Initializes the stream redirector."""
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf: str) -> None:
        """Redirects standard string writes to the logger."""
        for line in buf.rstrip().splitlines():
            log_entry: str = line.rstrip()
            if not log_entry:
                continue

            current_level: int = self.log_level
            if current_level == logging.ERROR:
                # Detect progress bars from downloads/uploads to prevent error logging
                is_progress_bar: bool = (
                    "%|" in log_entry or "B/s]" in log_entry or "it/s]" in log_entry
                )
                if is_progress_bar:
                    current_level = logging.INFO

            self.logger.log(current_level, log_entry)

    def flush(self) -> None:
        """Flush method for compatibility with file-like objects."""
        pass


class ProjectLogger:
    """Ensures that all generic output and exceptions are written to a file."""

    __instance: "ProjectLogger | None" = None

    def __init__(self, log_file: str = "logs.txt", level: int = logging.INFO) -> None:
        """Configures the root logger and redirects standard streams."""
        if ProjectLogger.__instance is not None:
            return

        self._configure_logging(log_file, level)
        self._redirect_std_streams()
        self._setup_excepthook()

        ProjectLogger.__instance = self

    def _configure_logging(self, log_file: str, level: int) -> None:
        """Sets up the root logger with a file handler and a console handler."""
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

        # Silence third-party warnings to avoid them showing up as ERRORs in the log files
        logging.getLogger("mlflow").setLevel(logging.ERROR)

    def _redirect_std_streams(self) -> None:
        """Redirects stdout and stderr to the configured logger."""
        sys.stdout = StreamToLogger(self.logger, logging.INFO)
        sys.stderr = StreamToLogger(self.logger, logging.ERROR)

    def _setup_excepthook(self) -> None:
        """Registers a global exception handler to log fatal errors."""

        def handle_exception(
            exc_type: type[BaseException],
            exc_value: BaseException,
            exc_traceback: types.TracebackType | None,
        ) -> None:
            """Internal handler for sys.excepthook."""
            if issubclass(exc_type, KeyboardInterrupt):
                self.logger.info("Execution interrupted by user (KeyboardInterrupt)")
                return

            self.logger.critical(
                "Uncaught exception - Program termination",
                exc_info=(exc_type, exc_value, exc_traceback),
            )

        sys.excepthook = handle_exception

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Returns a logger instance with the specified name."""
        return logging.getLogger(name)

    @classmethod
    def initialize(cls, log_file: str = "logs.txt") -> None:
        """Initializes the singleton logger system."""
        if cls.__instance is None:
            cls(log_file=log_file)
