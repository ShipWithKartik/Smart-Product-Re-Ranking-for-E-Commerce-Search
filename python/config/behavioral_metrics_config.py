"""
Configuration file for behavioral metric calculations
Provides configurable parameters for all behavioral metrics processing
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
from pathlib import Path


@dataclass
class ValidationConfig:
    """Configuration for data validation"""
    strict_mode: bool = True
    filter_invalid_rows: bool = True
    min_events_per_item: int = 1
    min_views_for_conversion: int = 1
    max_processing_time_minutes: int = 30
    required_columns: List[str] = field(default_factory=lambda: ['timestamp', 'visitorid', 'event', 'itemid'])
    valid_event_types: List[str] = field(default_factory=lambda: ['view', 'addtocart', 'transaction'])


@dataclass
class MetricsCalculationConfig:
    """Configuration for behavioral metrics calculation"""
    # Conversion rate calculation settings
    min_views_for_addtocart_rate: int = 1
    min_views_for_conversion_rate: int = 1
    min_addtocarts_for_cart_conversion: int = 1
    
    # Popularity scoring settings
    popularity_method: str = "view_based"  # Options: "view_based", "engagement_based", "weighted"
    popularity_weights: Dict[str, float] = field(default_factory=lambda: {
        'views': 0.4,
        'addtocarts': 0.3,
        'transactions': 0.3
    })
    
    # Temporal feature settings
    enable_temporal_features: bool = True
    temporal_window_hours: int = 24
    max_time_to_action_hours: int = 168  # 1 week
    
    # Engagement scoring settings
    engagement_decay_factor: float = 0.1
    repeat_visitor_bonus: float = 1.2
    
    # Outlier handling
    remove_outliers: bool = True
    outlier_method: str = "iqr"  # Options: "iqr", "zscore", "percentile"
    outlier_threshold: float = 3.0
    percentile_bounds: tuple = (0.01, 0.99)


@dataclass
class ProcessingConfig:
    """Configuration for data processing"""
    chunk_size: int = 10000
    memory_limit_mb: int = 1000
    parallel_processing: bool = False
    n_workers: int = 4
    
    # Feature engineering settings
    enable_temporal_features: bool = True
    enable_engagement_features: bool = True
    enable_composite_features: bool = True
    
    # Aggregation settings
    aggregation_level: str = "item"  # Options: "item", "category", "user_item"
    time_granularity: str = "day"  # Options: "hour", "day", "week"


@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling"""
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    continue_on_warnings: bool = True
    save_error_logs: bool = True
    log_level: str = "INFO"  # Options: "DEBUG", "INFO", "WARNING", "ERROR"
    
    # Memory management
    enable_memory_monitoring: bool = True
    memory_warning_threshold_mb: int = 500
    memory_error_threshold_mb: int = 1000


@dataclass
class OutputConfig:
    """Configuration for output settings"""
    save_intermediate_results: bool = True
    output_directory: str = "behavioral_metrics_output"
    file_format: str = "csv"  # Options: "csv", "parquet", "json"
    compression: Optional[str] = None  # Options: None, "gzip", "bz2"
    
    # File naming
    include_timestamp: bool = True
    file_prefix: str = "behavioral_metrics"
    
    # Summary reporting
    generate_summary_report: bool = True
    include_visualizations: bool = False


@dataclass
class BehavioralMetricsConfig:
    """Complete configuration for behavioral metrics processing"""
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    metrics: MetricsCalculationConfig = field(default_factory=MetricsCalculationConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    error_handling: ErrorHandlingConfig = field(default_factory=ErrorHandlingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'validation': self.validation.__dict__,
            'metrics': self.metrics.__dict__,
            'processing': self.processing.__dict__,
            'error_handling': self.error_handling.__dict__,
            'output': self.output.__dict__
        }
    
    def save_to_file(self, file_path: str) -> None:
        """Save configuration to JSON file"""
        config_dict = self.to_dict()
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'BehavioralMetricsConfig':
        """Load configuration from JSON file"""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            validation=ValidationConfig(**config_dict.get('validation', {})),
            metrics=MetricsCalculationConfig(**config_dict.get('metrics', {})),
            processing=ProcessingConfig(**config_dict.get('processing', {})),
            error_handling=ErrorHandlingConfig(**config_dict.get('error_handling', {})),
            output=OutputConfig(**config_dict.get('output', {}))
        )
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """Update configuration from dictionary"""
        for section, values in updates.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)


# Predefined configuration profiles
class ConfigProfiles:
    """Predefined configuration profiles for different use cases"""
    
    @staticmethod
    def development() -> BehavioralMetricsConfig:
        """Configuration for development/testing"""
        config = BehavioralMetricsConfig()
        config.processing.chunk_size = 1000
        config.validation.strict_mode = False
        config.error_handling.continue_on_warnings = True
        config.output.save_intermediate_results = True
        return config
    
    @staticmethod
    def production() -> BehavioralMetricsConfig:
        """Configuration for production use"""
        config = BehavioralMetricsConfig()
        config.processing.chunk_size = 50000
        config.validation.strict_mode = True
        config.error_handling.continue_on_warnings = False
        config.output.save_intermediate_results = False
        config.output.compression = "gzip"
        return config
    
    @staticmethod
    def large_dataset() -> BehavioralMetricsConfig:
        """Configuration for large datasets"""
        config = BehavioralMetricsConfig()
        config.processing.chunk_size = 100000
        config.processing.parallel_processing = True
        config.processing.memory_limit_mb = 2000
        config.error_handling.memory_error_threshold_mb = 2000
        config.output.file_format = "parquet"
        config.output.compression = "gzip"
        return config
    
    @staticmethod
    def minimal() -> BehavioralMetricsConfig:
        """Minimal configuration for basic processing"""
        config = BehavioralMetricsConfig()
        config.processing.enable_temporal_features = False
        config.processing.enable_engagement_features = False
        config.processing.enable_composite_features = False
        config.metrics.enable_temporal_features = False
        config.output.save_intermediate_results = False
        config.output.generate_summary_report = False
        return config


# Default configuration instance
DEFAULT_CONFIG = BehavioralMetricsConfig()

# Configuration validation
def validate_config(config: BehavioralMetricsConfig) -> List[str]:
    """
    Validate configuration settings and return list of issues
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    # Validate chunk size
    if config.processing.chunk_size <= 0:
        errors.append("Processing chunk_size must be positive")
    
    # Validate memory limits
    if config.processing.memory_limit_mb <= 0:
        errors.append("Memory limit must be positive")
    
    # Validate thresholds
    if config.metrics.min_views_for_conversion_rate < 0:
        errors.append("min_views_for_conversion_rate cannot be negative")
    
    # Validate popularity weights
    if config.metrics.popularity_method == "weighted":
        weights = config.metrics.popularity_weights
        if not (0.99 <= sum(weights.values()) <= 1.01):
            errors.append("Popularity weights must sum to approximately 1.0")
    
    # Validate percentile bounds
    lower, upper = config.metrics.percentile_bounds
    if not (0 <= lower < upper <= 1):
        errors.append("Percentile bounds must be between 0 and 1, with lower < upper")
    
    # Validate file format
    valid_formats = ["csv", "parquet", "json"]
    if config.output.file_format not in valid_formats:
        errors.append(f"File format must be one of: {valid_formats}")
    
    return errors


# Configuration utilities
def create_config_file(profile: str = "default", output_path: str = "behavioral_metrics_config.json") -> None:
    """
    Create a configuration file with specified profile
    
    Args:
        profile: Configuration profile name
        output_path: Path to save configuration file
    """
    if profile == "development":
        config = ConfigProfiles.development()
    elif profile == "production":
        config = ConfigProfiles.production()
    elif profile == "large_dataset":
        config = ConfigProfiles.large_dataset()
    elif profile == "minimal":
        config = ConfigProfiles.minimal()
    else:
        config = DEFAULT_CONFIG
    
    config.save_to_file(output_path)
    print(f"Configuration file created: {output_path}")


def load_config_with_validation(file_path: str) -> BehavioralMetricsConfig:
    """
    Load configuration from file with validation
    
    Args:
        file_path: Path to configuration file
        
    Returns:
        Validated configuration object
        
    Raises:
        ValueError: If configuration is invalid
    """
    config = BehavioralMetricsConfig.load_from_file(file_path)
    
    errors = validate_config(config)
    if errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    return config


if __name__ == "__main__":
    # Example usage - create configuration files for different profiles
    profiles = ["default", "development", "production", "large_dataset", "minimal"]
    
    for profile in profiles:
        output_file = f"behavioral_metrics_config_{profile}.json"
        create_config_file(profile, output_file)
        print(f"Created {output_file}")
    
    # Demonstrate configuration validation
    config = DEFAULT_CONFIG
    errors = validate_config(config)
    if errors:
        print(f"Configuration errors: {errors}")
    else:
        print("Default configuration is valid")