import logging
import logging.handlers
import os
from datetime import datetime
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
import json

# Custom theme for Rich console
CUSTOM_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red bold",
    "debug": "grey50",
    "critical": "red bold reverse"
})

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record):
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage()
        }
        
        if hasattr(record, 'props'):
            log_obj.update(record.props)
            
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_obj)

class StructuredLogger(logging.Logger):
    """Custom logger class that supports structured logging."""
    
    def _log_with_props(self, level, msg, props=None, *args, **kwargs):
        """Log a message with additional properties."""
        if props:
            extra = kwargs.get('extra', {})
            extra['props'] = props
            kwargs['extra'] = extra
        self._log(level, msg, args, **kwargs)
    
    def info_with_props(self, msg, props=None, *args, **kwargs):
        """Log info message with properties."""
        self._log_with_props(logging.INFO, msg, props, *args, **kwargs)
    
    def error_with_props(self, msg, props=None, *args, **kwargs):
        """Log error message with properties."""
        self._log_with_props(logging.ERROR, msg, props, *args, **kwargs)
    
    def debug_with_props(self, msg, props=None, *args, **kwargs):
        """Log debug message with properties."""
        self._log_with_props(logging.DEBUG, msg, props, *args, **kwargs)

class LogManager:
    """Manages logging configuration and setup."""
    
    def __init__(self, app_name: str = "pdf_chat"):
        self.app_name = app_name
        self.console = Console(theme=CUSTOM_THEME)
        self.log_dir = "logs"
        self.ensure_log_directory()
        
    def ensure_log_directory(self):
        """Ensure log directory exists."""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
    def get_log_file_path(self, log_type: str) -> str:
        """Generate log file path based on type and date."""
        timestamp = datetime.now().strftime("%Y%m%d")
        return os.path.join(self.log_dir, f"{self.app_name}_{log_type}_{timestamp}.log")
    
    def setup_logging(self, 
                     debug_mode: bool = False, 
                     log_to_file: bool = True,
                     console_format: Optional[str] = None) -> None:
        """
        Set up logging configuration.
        
        Args:
            debug_mode: Enable debug logging
            log_to_file: Enable file logging
            console_format: Optional custom console format
        """
        # Register custom logger class
        logging.setLoggerClass(StructuredLogger)
        
        # Base configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Console handler with Rich formatting
        console_handler = RichHandler(
            console=self.console,
            show_path=debug_mode,
            enable_link_path=debug_mode,
            markup=True
        )
        console_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)
        
        # Set console format
        if console_format:
            console_handler.setFormatter(logging.Formatter(console_format))
        else:
            console_handler.setFormatter(logging.Formatter(
                '%(message)s' if not debug_mode else '%(asctime)s - %(levelname)s - %(message)s'
            ))
            
        root_logger.addHandler(console_handler)
        
        if log_to_file:
            # JSON formatted file handler for structured logging
            json_handler = logging.handlers.RotatingFileHandler(
                self.get_log_file_path("structured"),
                maxBytes=10_000_000,  # 10MB
                backupCount=5
            )
            json_handler.setFormatter(JSONFormatter())
            json_handler.setLevel(logging.DEBUG)
            root_logger.addHandler(json_handler)
            
            # Regular file handler for traditional logging
            file_handler = logging.handlers.RotatingFileHandler(
                self.get_log_file_path("regular"),
                maxBytes=10_000_000,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            file_handler.setLevel(logging.DEBUG)
            root_logger.addHandler(file_handler)

def get_logger(name: str) -> StructuredLogger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)

# Example usage and helper functions
def log_system_info(logger: StructuredLogger) -> None:
    """Log system information."""
    import platform
    import psutil
    
    system_info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available
    }
    
    logger.info_with_props(
        "System information logged",
        props=system_info
    )

def log_operation_metrics(logger: StructuredLogger, 
                         operation: str, 
                         duration: float, 
                         success: bool, 
                         **metrics) -> None:
    """Log operation metrics."""
    logger.info_with_props(
        f"Operation {operation} completed",
        props={
            'operation': operation,
            'duration_seconds': duration,
            'success': success,
            'metrics': metrics
        }
    )

# Example initialization code
if __name__ == "__main__":
    # Initialize logging
    log_manager = LogManager()
    log_manager.setup_logging(debug_mode=True)
    
    # Get logger instance
    logger = get_logger(__name__)
    
    # Example logging
    logger.info("Application started")
    log_system_info(logger)
    
    try:
        # Simulate some operation
        logger.debug("Starting operation")
        # ... operation code ...
        log_operation_metrics(logger, "example_operation", 1.23, True, 
                            items_processed=100, errors=0)
    except Exception as e:
        logger.error_with_props(
            "Operation failed",
            props={'error_type': type(e).__name__, 'error_msg': str(e)}
        )