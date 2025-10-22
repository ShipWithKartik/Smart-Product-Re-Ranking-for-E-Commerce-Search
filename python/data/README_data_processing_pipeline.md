# Data Processing Pipeline Documentation

## Overview

The data processing pipeline provides a comprehensive, error-resilient system for processing e-commerce event data and calculating behavioral metrics. This finalized implementation includes robust error handling, configurable processing options, and extensive monitoring capabilities.

## Components

### 1. EventDataProcessor (`data_processing_pipeline.py`)

The main pipeline class that orchestrates the complete data processing workflow with comprehensive error handling.

**Key Features:**
- Robust error handling with retry logic
- Memory usage monitoring
- Processing time tracking
- Configurable validation and processing options
- Automatic result saving with error logs

**Usage:**
```python
from data.data_processing_pipeline import EventDataProcessor, process_events_data

# Simple usage
results = process_events_data("events.csv")

# Advanced usage with custom configuration
config = {
    'validation': {'strict_mode': True, 'filter_invalid_rows': True},
    'processing': {'enable_temporal_features': True, 'chunk_size': 10000},
    'error_handling': {'max_retries': 3, 'continue_on_warnings': True}
}

processor = EventDataProcessor(config)
results = processor.run_complete_pipeline("events.csv", "output_directory")
```

### 2. Configuration System (`config/behavioral_metrics_config.py`)

Comprehensive configuration system for all aspects of behavioral metrics processing.

**Configuration Sections:**
- **ValidationConfig**: Data validation settings
- **MetricsCalculationConfig**: Behavioral metrics calculation parameters
- **ProcessingConfig**: Data processing and feature engineering settings
- **ErrorHandlingConfig**: Error handling and retry logic
- **OutputConfig**: File output and reporting settings

**Predefined Profiles:**
- `development`: Fast processing for development/testing
- `production`: Optimized settings for production use
- `large_dataset`: Configuration for processing large datasets
- `minimal`: Basic processing with minimal features

**Usage:**
```python
from config.behavioral_metrics_config import ConfigProfiles, BehavioralMetricsConfig

# Use predefined profile
config = ConfigProfiles.production()

# Create custom configuration
config = BehavioralMetricsConfig()
config.processing.chunk_size = 50000
config.metrics.enable_temporal_features = True
config.save_to_file("my_config.json")

# Load from file
config = BehavioralMetricsConfig.load_from_file("my_config.json")
```

### 3. Data Processing Utilities (`utils/data_processing_utilities.py`)

Collection of utility classes and functions for common data processing tasks.

**Utility Classes:**
- **DataValidator**: Schema and data quality validation
- **MemoryMonitor**: Memory usage monitoring and alerts
- **ProcessingTimer**: Performance timing and checkpoints
- **DataCleaner**: Data cleaning and outlier handling
- **FileManager**: File I/O operations and directory management
- **ProgressTracker**: Progress tracking for long operations

**Usage:**
```python
from utils.data_processing_utilities import DataValidator, MemoryMonitor, DataCleaner

# Validate data
validator = DataValidator()
validation_result = validator.validate_events_schema(events_df)

# Monitor memory
memory_monitor = MemoryMonitor(warning_threshold_mb=500)
memory_status = memory_monitor.check_memory_usage()

# Clean data
cleaner = DataCleaner()
cleaned_df = cleaner.clean_events_data(events_df, remove_duplicates=True)
```

## Sample Processing Scripts

### 1. Sample Data Processing Demo (`examples/sample_data_processing_demo.py`)

Comprehensive demonstration script that shows all pipeline capabilities using generated sample data.

**Features:**
- Generates realistic sample event data
- Demonstrates basic and advanced processing
- Shows error handling capabilities
- Compares different configuration profiles

**Run the demo:**
```bash
cd python/examples
python sample_data_processing_demo.py
```

### 2. Behavioral Metrics Example (`examples/behavioral_metrics_example.py`)

Real-world example using actual event data to calculate and analyze behavioral metrics.

**Features:**
- Loads and processes real event data
- Performs detailed behavioral analysis
- Generates insights and recommendations
- Creates comprehensive reports

**Run the example:**
```bash
cd python/examples
python behavioral_metrics_example.py
```

## Configuration Files

The system automatically creates configuration files for different use cases:

### Default Configuration Files
- `behavioral_metrics_config_default.json`: Standard configuration
- `behavioral_metrics_config_development.json`: Development settings
- `behavioral_metrics_config_production.json`: Production-optimized settings
- `behavioral_metrics_config_large_dataset.json`: Large dataset processing
- `behavioral_metrics_config_minimal.json`: Minimal processing

### Creating Custom Configurations

```python
from config.behavioral_metrics_config import create_config_file

# Create configuration file for specific profile
create_config_file("production", "my_production_config.json")

# Create custom configuration
config = BehavioralMetricsConfig()
config.processing.chunk_size = 25000
config.metrics.min_views_for_conversion_rate = 10
config.save_to_file("custom_config.json")
```

## Error Handling

The pipeline includes comprehensive error handling:

### Error Types Handled
- **File I/O Errors**: Missing files, permission issues
- **Data Validation Errors**: Invalid schemas, corrupt data
- **Memory Errors**: Out of memory conditions
- **Processing Errors**: Calculation failures, timeout issues

### Error Recovery
- Automatic retry with exponential backoff
- Graceful degradation (continue with warnings)
- Detailed error logging and reporting
- Intermediate result saving on failure

### Error Logging
```python
# Error logs are automatically saved to:
# - processing_log.json (processing steps and status)
# - error_log.json (detailed error information on failure)
```

## Performance Monitoring

### Memory Monitoring
```python
# Automatic memory monitoring with configurable thresholds
config.error_handling.memory_warning_threshold_mb = 500
config.error_handling.memory_error_threshold_mb = 1000
```

### Processing Time Tracking
```python
# Automatic timing of all processing steps
# Results include:
# - Total processing time
# - Individual step timings
# - Performance checkpoints
```

### Progress Tracking
```python
# For large datasets, progress is logged every 10 seconds
# Includes:
# - Items processed
# - Percentage complete
# - Estimated time remaining
```

## Output Files

The pipeline generates several output files:

### Data Files
- `processed_events.csv`: Cleaned and validated event data
- `behavioral_metrics.csv`: Calculated behavioral metrics
- `feature_matrix.csv`: Enhanced feature matrix (if features enabled)

### Reports and Logs
- `processing_log.json`: Detailed processing log
- `pipeline_summary.json`: Summary statistics and metadata
- `behavioral_insights.txt`: Human-readable insights report

### Configuration Files
- Configuration files for different processing profiles
- Custom configuration files as needed

## Best Practices

### For Development
```python
# Use development profile for faster iteration
config = ConfigProfiles.development()
config.processing.chunk_size = 1000  # Small chunks for testing
config.validation.strict_mode = False  # Allow some data issues
```

### For Production
```python
# Use production profile for reliability
config = ConfigProfiles.production()
config.validation.strict_mode = True  # Strict validation
config.error_handling.continue_on_warnings = False  # Fail on issues
config.output.compression = "gzip"  # Compress output files
```

### For Large Datasets
```python
# Use large dataset profile for performance
config = ConfigProfiles.large_dataset()
config.processing.chunk_size = 100000  # Large chunks
config.processing.parallel_processing = True  # Enable parallelization
config.output.file_format = "parquet"  # Efficient format
```

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce chunk_size in configuration
   - Enable chunked processing
   - Use parquet format for large files

2. **Validation Errors**
   - Check data schema matches requirements
   - Enable filter_invalid_rows to skip bad data
   - Review validation error messages in logs

3. **Performance Issues**
   - Increase chunk_size for better performance
   - Disable unnecessary features (temporal, engagement)
   - Use minimal configuration profile

4. **File I/O Errors**
   - Check file permissions
   - Ensure sufficient disk space
   - Verify file paths are correct

### Debug Mode
```python
# Enable detailed logging for debugging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use development configuration with detailed logging
config = ConfigProfiles.development()
config.error_handling.log_level = "DEBUG"
```

## Integration Examples

### With Existing Pipeline
```python
# Integrate with existing behavioral metrics pipeline
from data.behavioral_pipeline import BehavioralMetricsPipeline
from data.data_processing_pipeline import EventDataProcessor

# Use enhanced processor in existing pipeline
processor = EventDataProcessor()
events_df = processor.load_events_with_error_handling("events.csv")
metrics_df = processor.calculate_behavioral_metrics_with_error_handling()
```

### With Machine Learning Pipeline
```python
# Prepare data for ML model training
results = process_events_data("events.csv", output_dir="ml_input")
features_df = results['features_df']

# Features are ready for model training
from models.model_training_pipeline import ModelTrainer
trainer = ModelTrainer()
model = trainer.train_model(features_df)
```

This finalized data processing pipeline provides a robust, configurable, and well-documented system for processing e-commerce event data with comprehensive error handling and monitoring capabilities.