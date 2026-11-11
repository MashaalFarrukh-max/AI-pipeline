
"""Configure structured JSON logging for the RAG API."""

import json
import logging
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Format log records as structured JSON entries."""

    def format(self, record: logging.LogRecord) -> str:
        """Convert a log record into a JSON-formatted string.

        Args:
            record: Logging record containing the event details.

        Returns:
            A JSON string containing the timestamp, log level,
            message, and any available request information.
        """
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }

        if hasattr(record, "request_method"):
            log_entry["method"] = record.request_method

        if hasattr(record, "request_path"):
            log_entry["path"] = record.request_path

        if hasattr(record, "status_code"):
            log_entry["status_code"] = record.status_code

        if hasattr(record, "duration_ms"):
            log_entry["duration_ms"] = record.duration_ms

        return json.dumps(log_entry)


def setup_logging() -> logging.Logger:
    """Set up and return the RAG API logger.

    Returns:
        A configured logger with JSON-formatted console output.
    """
    logger = logging.getLogger("rag_api")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    return logger
