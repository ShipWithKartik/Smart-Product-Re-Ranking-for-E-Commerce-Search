"""
Comprehensive logging and error handling configuration for the Smart Product Re-Ranking system
Provides centralized logging, error handling, and monitoring capabilities
"""

import logging
import logging.handlers
import sys
import traceback
import functools
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Union
from datetime import datetime
import json
import os


class SmartRankingLogger:
    """
    Centralized logger for the Smart Product Re-Ranking system
    Provides structured logging with different levels and output formats
    """
    
    def __init__(self, 
                 name: str = "smart_ranking",
                 log_level: str = "INFO",
                 log_dir: str = "logs",
                 enable_file_logging: bool = True,
                 enable_console_logging: bool = True,
                 max_file_size_mb: int = 10,
                 backup_count: int = 5):
        """
        Initialize the logger
        
        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files
            enable_file_logging: Whether to log to files
            enable_console_logging: Whether to log to console
            max_file_size_mb: Maximum size of log files in MB
            backup_count: Number of backup log files to keep
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        self.detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        self.simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup handlers
        if enable_console_logging:
            self._setup_console_handler()
        
        if enable_file_logging:
            self._setup_file_handlers(max_file_size_mb, backup_count)
    
    def _setup_console_handler(self):
        """Setup console logging handler"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.simple_formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handlers(self, max_file_size_mb: int, backup_count: int):
        """Setup file logging handlers"""
        # General log file
        general_log_file = self.log_dir / f"{self.name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            general_log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(self.detailed_formatter)
        self.logger.addHandler(file_handler)
        
        # Error log file
        error_log_file = self.log_dir / f"{self.name}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(self.detailed_formatter)
        self.logger.addHandler(error_handler)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message"""
        self.logger.debug(message, extra=extra or {})
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message"""
        self.logger.info(message, extra=extra or {})
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        self.logger.warning(message, extra=extra or {})
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """Log error message"""
        self.logger.error(message, extra=extra or {}, exc_info=exc_info)
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """Log critical message"""
        self.logger.critical(message, extra=extra or {}, exc_info=exc_info)
    
    def log_performance(self, operation: str, duration: float, extra: Optional[Dict[str, Any]] = None):
        """Log performance metrics"""
        perf_data = {
            'operation': operation,
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat()
        }
        if extra:
            perf_data.update(extra)
        
        self.info(f"Performance: {operation} completed in {duration:.3f}s", extra=perf_data)
    
    def log_data_quality(self, dataset: str, metrics: Dict[str, Any]):
        """Log data quality metrics"""
        quality_data = {
            'dataset': dataset,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        self.info(f"Data Quality: {dataset}", extra=quality_data)
    
    def log_model_performance(self, model_name: str, metrics: Dict[str, float]):
        """Log model performance metrics"""
        perf_data = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        self.info(f"Model Performance: {model_name}", extra=perf_data)


class ErrorHandler:
    """
    Centralized error handling for the Smart Product Re-Ranking system
    Provides structured error handling, recovery strategies, and error reporting
    """
    
    def __init__(self, logger: SmartRankingLogger):
        self.logger = logger
        self.error_counts = {}
        self.error_history = []
    
    def handle_error(self, 
                    error: Exception, 
                    context: str = "",
                    recoverable: bool = True,
                    retry_count: int = 0,
                    max_retries: int = 3) -> bool:
        """
        Handle an error with appropriate logging and recovery
        
        Args:
            error: The exception that occurred
            context: Context information about where the error occurred
            recoverable: Whether the error is recoverable
            retry_count: Current retry attempt
            max_retries: Maximum number of retries
            
        Returns:
            True if error was handled and operation can continue, False otherwise
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        # Track error frequency
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Log error details
        error_data = {
            'error_type': error_type,
            'error_message': error_message,
            'context': context,
            'recoverable': recoverable,
            'retry_count': retry_count,
            'max_retries': max_retries,
            'traceback': traceback.format_exc()
        }
        
        # Store in error history
        self.error_history.append({
            'timestamp': datetime.now().isoformat(),
            **error_data
        })
        
        # Log based on severity
        if retry_count >= max_retries or not recoverable:
            self.logger.critical(
                f"Critical Error in {context}: {error_message}",
                extra=error_data,
                exc_info=True
            )
            return False
        else:
            self.logger.warning(
                f"Recoverable Error in {context}: {error_message} (Retry {retry_count}/{max_retries})",
                extra=error_data
            )
            return True
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered"""
        return {
            'total_errors': len(self.error_history),
            'error_counts_by_type': self.error_counts,
            'recent_errors': self.error_history[-10:] if self.error_history else []
        }


def with_error_handling(logger: SmartRankingLogger, 
                       context: str = "",
                       max_retries: int = 3,
                       retry_delay: float = 1.0,
                       recoverable_exceptions: tuple = (Exception,)):
    """
    Decorator for adding error handling to functions
    
    Args:
        logger: Logger instance
        context: Context description for error logging
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        recoverable_exceptions: Tuple of exception types that are recoverable
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = ErrorHandler(logger)
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except recoverable_exceptions as e:
                    is_recoverable = attempt < max_retries
                    
                    if error_handler.handle_error(
                        e, 
                        context or func.__name__,
                        is_recoverable,
                        attempt,
                        max_retries
                    ):
                        if attempt < max_retries:
                            time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                            continue
                    
                    # If we get here, error was not recoverable or max retries exceeded
                    raise
                except Exception as e:
                    # Non-recoverable exception
                    error_handler.handle_error(e, context or func.__name__, False)
                    raise
            
        return wrapper
    return decorator


def with_performance_logging(logger: SmartRankingLogger, operation_name: str = ""):
    """
    Decorator for adding performance logging to functions
    
    Args:
        logger: Logger instance
        operation_name: Name of the operation for logging
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            operation = operation_name or func.__name__
            
            try:
                logger.debug(f"Starting operation: {operation}")
                result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                logger.log_performance(operation, duration)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Operation {operation} failed after {duration:.3f}s: {str(e)}",
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


class SystemMonitor:
    """
    System monitoring for resource usage and performance
    """
    
    def __init__(self, logger: SmartRankingLogger):
        self.logger = logger
        self.start_time = time.time()
        self.checkpoints = []
    
    def checkpoint(self, name: str, extra_data: Optional[Dict[str, Any]] = None):
        """Create a performance checkpoint"""
        current_time = time.time()
        checkpoint_data = {
            'checkpoint_name': name,
            'elapsed_time': current_time - self.start_time,
            'timestamp': datetime.now().isoformat()
        }
        
        if extra_data:
            checkpoint_data.update(extra_data)
        
        self.checkpoints.append(checkpoint_data)
        self.logger.debug(f"Checkpoint: {name}", extra=checkpoint_data)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        total_time = time.time() - self.start_time
        
        return {
            'total_execution_time': total_time,
            'num_checkpoints': len(self.checkpoints),
            'checkpoints': self.checkpoints,
            'average_checkpoint_interval': total_time / len(self.checkpoints) if self.checkpoints else 0
        }


# Global logger instance
_global_logger: Optional[SmartRankingLogger] = None


def get_logger(name: str = "smart_ranking") -> SmartRankingLogger:
    """Get or create global logger instance"""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = SmartRankingLogger(name)
    
    return _global_logger


def setup_logging(log_level: str = "INFO", 
                 log_dir: str = "logs",
                 enable_file_logging: bool = True) -> SmartRankingLogger:
    """
    Setup global logging configuration
    
    Args:
        log_level: Logging level
        log_dir: Directory for log files
        enable_file_logging: Whether to enable file logging
        
    Returns:
        Configured logger instance
    """
    global _global_logger
    
    _global_logger = SmartRankingLogger(
        log_level=log_level,
        log_dir=log_dir,
        enable_file_logging=enable_file_logging
    )
    
    return _global_logger


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logger = setup_logging("DEBUG", "test_logs")
    
    # Test basic logging
    logger.info("System initialized")
    logger.debug("Debug information")
    logger.warning("This is a warning")
    
    # Test performance logging
    @with_performance_logging(logger, "test_operation")
    def test_function():
        time.sleep(0.1)
        return "success"
    
    result = test_function()
    
    # Test error handling
    @with_error_handling(logger, "test_error_handling", max_retries=2)
    def failing_function():
        raise ValueError("Test error")
    
    try:
        failing_function()
    except ValueError:
        logger.info("Error handling test completed")
    
    # Test system monitoring
    monitor = SystemMonitor(logger)
    monitor.checkpoint("initialization")
    time.sleep(0.05)
    monitor.checkpoint("processing")
    
    summary = monitor.get_performance_summary()
    logger.info("Performance summary", extra=summary)
    
    print("Logging system test completed. Check test_logs directory for log files.")