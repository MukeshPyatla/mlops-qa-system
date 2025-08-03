"""
Logging utilities for the LLM-Powered Q&A System.
"""

import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import structlog
from structlog.stdlib import LoggerFactory

# Global logger instance
_logger: Optional[structlog.stdlib.BoundLogger] = None


def setup_logging(
    level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
    enable_console: bool = True
) -> None:
    """
    Setup structured logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format ("json" or "text")
        log_file: Optional log file path
        enable_console: Whether to enable console logging
    """
    global _logger
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout if enable_console else None,
        level=getattr(logging, level.upper()),
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(file_handler)
    
    # Create logger instance
    _logger = structlog.get_logger()


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Optional logger name
        
    Returns:
        Structured logger instance
    """
    if _logger is None:
        setup_logging()
    
    if name:
        return structlog.get_logger(name)
    return _logger


def log_function_call(func_name: str, **kwargs) -> structlog.stdlib.BoundLogger:
    """
    Create a logger with function call context.
    
    Args:
        func_name: Name of the function being called
        **kwargs: Additional context to log
        
    Returns:
        Logger with function call context
    """
    logger = get_logger()
    return logger.bind(function=func_name, **kwargs)


def log_performance(operation: str, duration: float, **kwargs) -> None:
    """
    Log performance metrics.
    
    Args:
        operation: Name of the operation
        duration: Duration in seconds
        **kwargs: Additional metrics to log
    """
    logger = get_logger()
    logger.info(
        "Performance metric",
        operation=operation,
        duration_seconds=duration,
        **kwargs
    )


def log_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an error with context.
    
    Args:
        error: The exception that occurred
        context: Optional context dictionary
    """
    logger = get_logger()
    error_context = {
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    
    if context:
        error_context.update(context)
    
    logger.error("Error occurred", **error_context, exc_info=True)


def log_system_health(metrics: Dict[str, Any]) -> None:
    """
    Log system health metrics.
    
    Args:
        metrics: Dictionary of health metrics
    """
    logger = get_logger()
    logger.info("System health check", **metrics)


# Convenience functions for common logging patterns
def log_data_collection(source: str, count: int, **kwargs) -> None:
    """Log data collection events."""
    logger = get_logger()
    logger.info("Data collected", source=source, count=count, **kwargs)


def log_model_operation(operation: str, model_name: str, **kwargs) -> None:
    """Log model operations."""
    logger = get_logger()
    logger.info("Model operation", operation=operation, model=model_name, **kwargs)


def log_api_request(method: str, endpoint: str, status_code: int, duration: float) -> None:
    """Log API request details."""
    logger = get_logger()
    logger.info(
        "API request",
        method=method,
        endpoint=endpoint,
        status_code=status_code,
        duration_seconds=duration
    ) 