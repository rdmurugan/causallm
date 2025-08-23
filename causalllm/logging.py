import logging
import os
import json
import sys
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path


class CausalLMLogger:
    """Centralized logging utility for CausalLM package."""
    
    _instance: Optional["CausalLMLogger"] = None
    _loggers: Dict[str, logging.Logger] = {}

    def __new__(cls) -> "CausalLMLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self.log_dir: Path = Path("logs")
            self.log_dir.mkdir(exist_ok=True)

    def get_logger(
        self,
        name: str,
        level: str = "INFO",
        log_to_file: bool = True,
        log_to_console: bool = True,
        json_format: bool = False
    ) -> logging.Logger:
        if name in self._loggers:
            return self._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.handlers.clear()

        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            formatter = (
                JSONFormatter() if json_format
                else logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        if log_to_file:
            log_file = self.log_dir / f"{name}.log"
            file_handler = logging.FileHandler(log_file)
            formatter = (
                JSONFormatter() if json_format
                else logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
                )
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        self._loggers[name] = logger
        return logger

    def setup_structured_logging(
        self,
        component: str,
        log_file: Optional[str] = None
    ) -> "StructuredLogger":
        if log_file is None:
            log_file = str(self.log_dir / f"{component}_structured.jsonl")
        return StructuredLogger(component, log_file)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        if hasattr(record, "extra_data"):
            log_entry.update(getattr(record, "extra_data"))

        return json.dumps(log_entry)


class StructuredLogger:
    """Logger for structured data (interactions, experiments, etc.)."""

    def __init__(self, component: str, log_file: str) -> None:
        self.component = component
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log_interaction(
        self,
        interaction_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "component": self.component,
            "interaction_type": interaction_type,
            "data": data,
        }

        if metadata:
            log_entry["metadata"] = metadata

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger = logging.getLogger(self.component)
            logger.error(f"Failed to write structured log: {e}")

    def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        error_data: Dict[str, Any] = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        self.log_interaction("error", error_data)

    def log_performance(
        self,
        operation: str,
        duration: float,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        perf_data: Dict[str, Any] = {
            "operation": operation,
            "duration_seconds": duration,
            "metrics": metrics or {}
        }
        self.log_interaction("performance", perf_data)


def get_logger(name: str, **kwargs: Any) -> logging.Logger:
    """Get a logger with default configuration."""
    return CausalLMLogger().get_logger(name, **kwargs)


def get_structured_logger(component: str, log_file: Optional[str] = None) -> StructuredLogger:
    """Get a structured logger for a component."""
    return CausalLMLogger().setup_structured_logging(component, log_file)


def setup_package_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_to_file: bool = True
) -> None:
    logger_manager = CausalLMLogger()
    components = [
        "causalllm.core",
        "causalllm.llm_client",
        "causalllm.counterfactual_engine",
        "causalllm.dag_parser",
        "causalllm.do_operator",
        "causalllm.scm_explainer",
        "causalllm.utils",
    ]

    for component in components:
        logger_manager.get_logger(
            component,
            level=level,
            json_format=json_format,
            log_to_file=log_to_file,
        )


class LogLevel:
    """Context manager to temporarily change log level."""

    def __init__(self, logger: logging.Logger, level: str) -> None:
        self.logger = logger
        self.new_level = getattr(logging, level.upper(), logging.INFO)
        self.old_level: Optional[int] = None

    def __enter__(self) -> logging.Logger:
        self.old_level = self.logger.level
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.old_level is not None:
            self.logger.setLevel(self.old_level)
