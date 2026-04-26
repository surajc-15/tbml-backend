"""
Reusable, industry-style logger utility for TBML services.

Usage:
    from utilities.logger import setup_logger, log_event

    logger = setup_logger("tbml.ingestion")
    log_event(logger, headline="Ingestion started", status="INFO", bank="bankc")
    log_event(logger, headline="Transaction flagged", status="SUCCESS", msg_id="TXN_123")
    log_event(logger, headline="Graph query failed", status="ERROR", error="Timeout")
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict

try:
    from colorama import init as colorama_init

    colorama_init()  # Enables ANSI colors on many Windows terminals.
except Exception:
    # Keep logger functional even if colorama is unavailable.
    pass


_STATUS_TO_LEVEL = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "SUCCESS": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_STATUS_COLORS = {
    "DEBUG": "\033[34m",     # Blue
    "INFO": "\033[36m",      # Cyan
    "SUCCESS": "\033[32m",   # Green
    "WARNING": "\033[33m",   # Yellow
    "WARN": "\033[33m",      # Yellow
    "ERROR": "\033[31m",     # Red
    "CRITICAL": "\033[35m",  # Magenta
}

_TEXT_COLORS = {
    "headline": "\033[1m",   # Bold
    "message": "\033[97m",   # Bright white
    "context_key": "\033[96m",   # Bright cyan
    "context_value": "\033[92m", # Bright green
    "error_value": "\033[91m",    # Bright red
}

_RESET_COLOR = "\033[0m"


class JsonFormatter(logging.Formatter):
    """Structured JSON formatter for production logging pipelines."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "logger": record.name,
            "level": record.levelname,
            "status": getattr(record, "status", record.levelname),
            "headline": getattr(record, "headline", record.getMessage()),
            "message": record.getMessage(),
        }

        # Attach custom context if present.
        context = getattr(record, "context", None)
        if context:
            payload["context"] = context

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Debug-friendly structured formatter with file/line and optional context."""

    def __init__(self, use_color: bool = True) -> None:
        super().__init__()
        self.use_color = use_color and self._supports_color()

    @staticmethod
    def _supports_color() -> bool:
        if not sys.stdout or not hasattr(sys.stdout, "isatty"):
            return False
        if not sys.stdout.isatty():
            return False
        if os.getenv("NO_COLOR"):
            return False
        return True

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        headline = getattr(record, "headline", record.getMessage())
        status = getattr(record, "status", record.levelname)
        context = getattr(record, "context", None)
        message = record.getMessage()

        color_prefix = ""
        color_suffix = ""
        if self.use_color:
            color_prefix = _STATUS_COLORS.get(str(status).upper(), "")
            color_suffix = _RESET_COLOR if color_prefix else ""

        status_text = f"{color_prefix}{status}{color_suffix}" if color_prefix else str(status)

        if self.use_color:
            headline_text = f"{_TEXT_COLORS['headline']}{headline}{_RESET_COLOR}"
            message_color = _STATUS_COLORS.get(str(status).upper(), _TEXT_COLORS["message"])
            message_text = f"{message_color}{message}{_RESET_COLOR}"
            context_text = self._format_context(context, status)
        else:
            headline_text = headline
            message_text = message
            context_text = f" | context={json.dumps(context, default=str)}" if context else ""

        return (
            f"{ts} | {status_text:<8} | {record.filename}:{record.lineno} | "
            f"{headline_text}"
            f" | {message_text}"
            f"{context_text}"
        )

    def _format_context(self, context: Any, status: str) -> str:
        if not context:
            return ""

        if isinstance(context, dict):
            parts = []
            for key, value in context.items():
                key_text = f"{_TEXT_COLORS['context_key']}{key}{_RESET_COLOR}"
                value_color = _TEXT_COLORS["error_value"] if str(key).lower() in {"error", "exception", "err"} else _TEXT_COLORS["context_value"]
                value_text = f"{value_color}{value}{_RESET_COLOR}"
                parts.append(f"{key_text}={value_text}")

            return " | context=" + ", ".join(parts)

        return f" | context={context}"


def setup_logger(
    name: str = "tbml",
    level: int = logging.INFO,
    use_json: bool = False,
    use_color: bool = True,
) -> logging.Logger:
    """Create or return a configured logger.

    Args:
        name: Logger name.
        level: Logging level (default INFO).
        use_json: If True, emit JSON logs; else plain text logs.
        use_color: If True and supported by terminal, colorize text logs by status.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times.
    if logger.handlers:
        return logger

    stream_handler = logging.StreamHandler(sys.stdout)

    if use_json:
        formatter = JsonFormatter()
    else:
        formatter = TextFormatter(use_color=use_color)

    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def log_event(
    logger: logging.Logger,
    headline: str,
    status: str = "INFO",
    message: str = "",
    **context: Any,
) -> None:
    """Standard event logger with headline + status + optional context.

    Args:
        logger: Configured logger object.
        headline: High-level event title.
        status: DEBUG | INFO | SUCCESS | WARNING | ERROR | CRITICAL.
        message: Detailed event message.
        **context: Arbitrary key-value metadata (msg_id, bank, amount, etc.).
    """
    normalized = (status or "INFO").upper()
    level = _STATUS_TO_LEVEL.get(normalized, logging.INFO)

    extra: Dict[str, Any] = {
        "headline": headline,
        "status": normalized,
        "context": context or None,
    }

    logger.log(level, message or headline, extra=extra)


def log_exception(
    logger: logging.Logger,
    headline: str,
    error: Exception,
    **context: Any,
) -> None:
    """Log exceptions with consistent structure and traceback."""
    extra: Dict[str, Any] = {
        "headline": headline,
        "status": "ERROR",
        "context": context or None,
    }
    logger.error(str(error), exc_info=True, extra=extra)
